from __future__ import annotations

import torch
import csv
from torch.utils.data import Dataset
from torch import Tensor
from pathlib import Path
from abc import ABC, abstractmethod


class URLDataset(Dataset, ABC):
    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    @abstractmethod
    def X(self) -> Tensor: ...

    @property
    @abstractmethod
    def y(self) -> Tensor: ...


class LabeledURLDataset(URLDataset):
    @staticmethod
    def from_csv(csv_path: Path, seq_len: int) -> LabeledURLDataset:
        urls = []
        labels = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for url, label in reader:
                urls.append(url)
                labels.append(not label.startswith("benign"))

        return LabeledURLDataset.from_str(urls, labels, seq_len)

    @staticmethod
    def from_str(
        urls: list[str], labels: list[bool], seq_len: int
    ) -> LabeledURLDataset:
        encoded_urls = [url.encode().ljust(seq_len, b"\x00")[:seq_len] for url in urls]
        X = torch.tensor([list(url) for url in encoded_urls], dtype=torch.int)
        y = torch.tensor(labels, dtype=torch.float)
        return LabeledURLDataset(X, y)

    def __init__(self, X: Tensor, y: Tensor) -> None:
        self._X = X
        self._y = y

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._X[index], self._y[index]

    def __len__(self) -> int:
        return len(self._y)

    @property
    def X(self) -> Tensor:
        return self._X

    @property
    def y(self) -> Tensor:
        return self._y


class FilteredURLDataset(URLDataset):
    def __init__(
        self, filter, threshold: float, batch_size: int, count: int, seq_len: int
    ):
        self.filter = filter
        self.threshold = threshold
        self.batch_size = batch_size
        self.count = count
        self.seq_len = seq_len
        self.reset()

    def reset(self):
        self._X = torch.randint(0, 256, (self.count, self.seq_len))
        self._y = self.filter.batched_contains(self.X)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._X[index], self._y[index]

    def __len__(self) -> int:
        return self.count

    @property
    def X(self) -> Tensor:
        return self._X

    @property
    def y(self) -> Tensor:
        return self._y
