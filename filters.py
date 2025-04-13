from __future__ import annotations


import math
from rbloom import Bloom
import torch
from torch import Tensor
from model import URLClassifer
from blake3 import blake3
from data import URLDataset
from train import test


class NaorEylonFilter:
    def __init__(self, items: Tensor, fpr: float, key: bytes, bits: int) -> None:
        n = math.ceil(-(math.log(2) ** 2) * bits / math.log(fpr))
        # self.filter = Bloom(n, fpr, hash_func=self.hash)
        print("filter args", items.size(0), fpr)
        self.filter = Bloom(items.size(0), fpr, hash_func=self.hash)
        print(self.filter.size_in_bits)
        self.key = key
        self.fpr = fpr

        items = items.cpu()
        for item in items:
            self.filter.add(item)

    def hash(self, item: Tensor) -> int:
        data = item.cpu().numpy().tobytes()
        return int.from_bytes(blake3(data, key=self.key).digest()[:16], signed=True)

    def batched_contains(self, items: Tensor) -> Tensor:
        return torch.tensor([item in self.filter for item in items], dtype=torch.bool)

    def size_in_bytes(self) -> int:
        return self.filter.size_in_bits // 8


def find_threshold(
    model: URLClassifer,
    dataset: URLDataset,
    fpr: float,
    batch_size: int,
    num_workers: int,
    iterations: int = 2,  # 16,
) -> float:
    left = 0.0
    right = 1.0
    for _ in range(iterations):
        mid = (left + right) / 2
        # TODO: replace me.
        _, test_fpr, _, _ = test(
            model,
            dataset,
            batch_size=batch_size,
            threshold=mid,
            num_workers=num_workers,
        )
        if test_fpr > fpr:
            left = mid
        else:
            right = mid
    return left


class DowntownBodegaFilter:
    def __init__(
        self,
        items: Tensor,
        fpr: float,
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

        # P(self(x) is T | x is F)
        # = P(model(x) is F | x is F) * P(fn_filter(x) is T | x is F)
        #   + P(model(x) is T | x is F) * P(tp_filter(x) is T | x is F)
        # => fpr = model_fnr * fn_filter_fpr + model_fpr * tp_filter_fpr
        # => fpr = model_fnr * fn_filter_fpr + fpr * tp_filter_fpr          (threshold is set such that model_fpr = fpr)
        # => fpr = model_fnr * filter_fpr + fpr * filter_fpr                (worst-case is max(tp_filter_fpr, fn_filter_fpr))
        # => fpr / 2 * (model_fnr + fpr) = filter_fpr
        model_fnr = hints.logical_not().sum().item() / hints.size(0)
        filter_fpr = fpr / (2 * (model_fnr + fpr))
        print("model args", model_fnr, fpr, filter_fpr)
        self.fn_filter = NaorEylonFilter(
            fn, filter_fpr, fn_filter_key, int(170_000 * 0.9)
        )
        self.tp_filter = NaorEylonFilter(
            tp, filter_fpr, tp_filter_key, int(170_000 * 0.1)
        )

    def batched_contains(self, items: Tensor) -> Tensor:
        hints = self.batched_hints(items)
        return self.batched_contains_with_hints(items, hints)

    def batched_hints(self, items: Tensor) -> Tensor:
        items = items.cuda()
        with torch.no_grad():
            self.model.eval()
            hints = torch.zeros(items.size(0), device=items.device)
            for i in range(0, items.size(0), self.batch_size):
                hints[i : i + self.batch_size] = self.model(
                    items[i : i + self.batch_size]
                )
            hints = hints > self.threshold
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


def main(cfg):
    model = URLClassifer(
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        seq_len=cfg["seq_len"],
        dropout=0.0,
    )
    model.load_state_dict(torch.load(cfg["model_path"]))
    model = model.cuda()
    model.eval()

    dataset = URLDataset.from_csv(cfg["test_path"], cfg["seq_len"])
    threshold = find_threshold(
        model,
        dataset,
        cfg["fpr"],
        cfg["batch_size"],
        cfg["num_workers"],
    )
    print("threshold =", threshold)

    fp_key = b'B\xae\x1a\xc2\xbf\x8e\xb3\x94w\xdab\x8b"}/\xb5\x93Hrg \xedY\x9b5\xce\x85\xc7u#7\xb3'
    tp_key = b"JS\xf8\xed!\xc4\x84\xc7z\x0f\xad+k\x94\xe3E\xe8xG\xea\x1bRu\xa7\xe0\xd2d\xf8C\xf3\x0c\xad"
    filter = DowntownBodegaFilter(
        dataset.X[dataset.y > 0.99],
        cfg["fpr"],
        model,
        threshold,
        cfg["batch_size"],
        fp_key,
        tp_key,
    )
    print(filter.size_in_bytes())

    dbf_correct = 0
    dbf_total = 0
    dbf_p = 0
    fn_correct = 0
    fn_total = 0
    fn_p = 0
    tp_correct = 0
    tp_total = 0
    tp_p = 0
    for i in range(0, len(dataset.X), 1024):
        x = dataset.X[i : i + 1024]
        y = dataset.y[i : i + 1024] > 0.99

        hints = filter.batched_hints(x).cpu()
        px = x[hints]
        py = y[hints]
        nx = x[~hints]
        ny = y[~hints]

        dbf_correct += torch.sum(filter.batched_contains(x) == y)
        dbf_p += y.sum()
        dbf_total += len(x)

        fn_correct += torch.sum(filter.fn_filter.batched_contains(nx) == ny)
        fn_p += ny.sum()
        fn_total += len(nx)

        tp_correct += torch.sum(filter.tp_filter.batched_contains(px) == py)
        tp_p += py.sum()
        tp_total += len(px)

    print(
        f"DBF FPR: 1 - {dbf_correct}/{dbf_total} = {1 - dbf_correct / dbf_total}, Positive: {dbf_p}/{dbf_total}"
    )
    print(
        f"FN FPR: 1 - {fn_correct}/{fn_total} = {1 - fn_correct / fn_total}, Positive: {fn_p}/{fn_total}, Size: {filter.fn_filter.filter.approx_items}, Bits: {filter.fn_filter.filter.size_in_bits}"
    )
    print(
        f"TP FPR: 1 - {tp_correct}/{tp_total} = {1 - tp_correct / tp_total}, Positive: {tp_p}/{tp_total}, Size: {filter.tp_filter.filter.approx_items}, Bits: {filter.tp_filter.filter.size_in_bits}"
    )


if __name__ == "__main__":
    cfg = {
        # Dataset.
        "test_path": "data/data.csv",
        "seq_len": 128,
        "num_workers": 1,
        # Model.
        "hidden_dim": 8,
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.1,
        "threshold": 0.5,
        "model_path": "models/transformer.pt",
        # Testing.
        "batch_size": 1024,
        # Filter.
        "fpr": 0.05,
    }
    main(cfg)
