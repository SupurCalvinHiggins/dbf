from __future__ import annotations


from abc import ABC, abstractmethod
from rbloom import Bloom
import torch
from torch import Tensor
from model import URLClassifer
from blake3 import blake3
from train import predict
from typing import Optional


class Filter(ABC):
    @abstractmethod
    def batched_contains(self, items: Tensor) -> Tensor: ...

    @abstractmethod
    def size_in_bytes(self) -> int: ...


def hash_item(item: Tensor, key: Optional[bytes] = None) -> int:
    data = item.cpu().numpy().tobytes()
    return int.from_bytes(blake3(data, key=key).digest()[:16], signed=True)


class HashSetFilter(Filter):
    def __init__(self, items: Tensor) -> None:
        self.hashes = set([self.hash(item) for item in items])

    def hash(self, item: Tensor) -> int:
        return hash_item(item)

    def batched_add(self, items: Tensor) -> None:
        for item in items:
            self.hashes.add(self.hash(item))

    def batched_contains(self, items: Tensor) -> Tensor:
        return torch.tensor(
            [self.hash(item) in self.hashes for item in items], dtype=torch.bool
        )

    def size_in_bytes(self) -> int:
        raise NotImplementedError


class NaorEylonFilter(Filter):
    def __init__(self, items: Tensor, fpr: float, key: bytes) -> None:
        self.filter = Bloom(items.size(0), fpr, hash_func=self.hash)
        self.fpr = fpr
        self.key = key

        items = items.cpu()
        for item in items:
            self.filter.add(item)

    def hash(self, item: Tensor) -> int:
        return hash_item(item, self.key)

    def batched_contains(self, items: Tensor) -> Tensor:
        return torch.tensor([item in self.filter for item in items], dtype=torch.bool)

    def size_in_bytes(self) -> int:
        return self.filter.size_in_bits // 8


class DowntownBodegaFilter(Filter):
    def __init__(
        self,
        items: Tensor,
        fpr: float,
        tpr: float,
        model: URLClassifer,
        threshold: float,
        batch_size: int,
        fn_filter_key: bytes,
        tp_filter_key: bytes,
    ) -> None:
        self.fpr = fpr
        self.model = model
        self.threshold = threshold
        self.batch_size = batch_size

        hints = self.batched_hints(items).cpu()
        fn = items[~hints]
        tp = items[hints]

        tp_filter_fpr = 1 - tpr
        fn_filter_fpr = tpr * fpr / (1 - fpr)

        self.fn_filter = NaorEylonFilter(fn, fn_filter_fpr, fn_filter_key)
        self.tp_filter = NaorEylonFilter(tp, tp_filter_fpr, tp_filter_key)

    def batched_contains(self, items: Tensor) -> Tensor:
        hints = self.batched_hints(items)
        return self.batched_contains_with_hints(items, hints)

    def batched_hints(self, items: Tensor) -> Tensor:
        items = items.cuda()
        with torch.no_grad():
            hints = predict(self.model, items, self.batch_size) > self.threshold
        return hints

    def batched_contains_with_hints(self, items: Tensor, hints: Tensor) -> Tensor:
        items = items.cpu()
        hints = hints.cpu()
        result = torch.zeros(items.size(0), dtype=torch.bool)
        result[hints] = self.tp_filter.batched_contains(items[hints])
        result[~hints] = self.fn_filter.batched_contains(items[~hints])
        return result

    def size_in_bytes(self) -> int:
        return (
            self.model.size_in_bytes()
            + self.tp_filter.size_in_bytes()
            + self.fn_filter.size_in_bytes()
        )
