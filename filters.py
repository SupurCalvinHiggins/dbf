from __future__ import annotations


from abc import ABC, abstractmethod
from rbloom import Bloom
import torch
from torch import Tensor, neg
from model import URLClassifer
from blake3 import blake3
from data import URLDataset, LabeledURLDataset
from train import find_threshold, test, predict


class Filter(ABC):
    @abstractmethod
    def batched_contains(self, items: Tensor) -> Tensor: ...

    @abstractmethod
    def size_in_bytes(self) -> int: ...


class NaorEylonFilter(Filter):
    def __init__(self, items: Tensor, fpr: float, key: bytes) -> None:
        self.filter = Bloom(items.size(0), fpr, hash_func=self.hash)
        self.fpr = fpr
        self.key = key

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
        print(f"{threshold=}")
        print(f"{items.size(0)=}")

        hints = self.batched_hints(items).cpu()
        fn = items[~hints]
        tp = items[hints]
        print(f"{fn.size(0)=}, {tp.size(0)=}")

        # P(self(x) is T | x is F)
        # = P(model(x) is F | x is F) * P(fn_filter(x) is T | x is F)
        #   + P(model(x) is T | x is F) * P(tp_filter(x) is T | x is F)
        # => fpr = model_fnr * fn_filter_fpr + model_fpr * tp_filter_fpr
        # => fpr = model_fnr * fn_filter_fpr + fpr * tp_filter_fpr          (threshold is set such that model_fpr = fpr)
        # => fpr = model_fnr * filter_fpr + fpr * filter_fpr                (worst-case is max(tp_filter_fpr, fn_filter_fpr))
        # => fpr / (model_fnr + fpr) = filter_fpr
        # TODO: model fnr is drasitic overestimate because it is only considering the ppositive elements.
        # should not divide by hints.size(0) instead by the total size of the entire dataset
        # model_fnr = hints.logical_not().sum().item() / hints.size(0)

        # fpr = model_fnr * filter_fpr + model_tnr * filter_fpr
        # fpr = (1 - model_tnr) * filter_fpr + model_tnr * filter_fpr
        # print(f"{1-fpr=}, {tpr=}")
        tp_filter_fpr = 1 - tpr
        fn_filter_fpr = tpr * fpr / (1 - fpr)
        # print(f"{fn_filter_fpr=}, {tp_filter_fpr=}")
        # filter_fpr = fpr / (tnr + fpr)
        # print(f"{fpr=}, {tnr=}, {filter_fpr=}")
        self.fn_filter = NaorEylonFilter(fn, fn_filter_fpr, fn_filter_key)
        self.tp_filter = NaorEylonFilter(tp, tp_filter_fpr, tp_filter_key)

        assert self.fn_filter.batched_contains(fn).sum() == fn.size(0)
        assert self.tp_filter.batched_contains(tp).sum() == tp.size(0)

        # must always store all positives
        # required size
        #

    def batched_contains(self, items: Tensor) -> Tensor:
        hints = self.batched_hints(items)
        return self.batched_contains_with_hints(items, hints)

    def batched_hints(self, items: Tensor) -> Tensor:
        items = items.cuda()
        with torch.no_grad():
            return predict(self.model, items, self.batch_size) > self.threshold
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

    dataset = LabeledURLDataset.from_csv(cfg["test_path"], cfg["seq_len"])
    threshold = find_threshold(
        model,
        dataset,
        cfg["fpr"],
        cfg["batch_size"],
        verbose=True,
    )
    tpr, fpr, _, _ = test(
        model, dataset, cfg["batch_size"], threshold, num_workers=8, verbose=True
    )

    neg_x = dataset.X[dataset.y < 0.01]
    pos_x = dataset.X[dataset.y > 0.99]

    fp_key = b'B\xae\x1a\xc2\xbf\x8e\xb3\x94w\xdab\x8b"}/\xb5\x93Hrg \xedY\x9b5\xce\x85\xc7u#7\xb3'
    tp_key = b"JS\xf8\xed!\xc4\x84\xc7z\x0f\xad+k\x94\xe3E\xe8xG\xea\x1bRu\xa7\xe0\xd2d\xf8C\xf3\x0c\xad"

    filter = DowntownBodegaFilter(
        pos_x,
        fpr,
        tpr,
        model,
        threshold,
        cfg["batch_size"],
        fp_key,
        tp_key,
    )
    ne_filter = NaorEylonFilter(pos_x, cfg["fpr"], key=fp_key)

    filter_fp = 0
    tp_filter_fp = 0
    fn_filter_fp = 0
    ne_filter_fp = 0

    for i in range(0, len(neg_x), 1024):
        x = neg_x[i : i + 1024]

        filter_fp += filter.batched_contains(x).sum().item()
        tp_filter_fp += filter.tp_filter.batched_contains(x).sum().item()
        fn_filter_fp += filter.fn_filter.batched_contains(x).sum().item()
        ne_filter_fp += ne_filter.batched_contains(x).sum().item()

    filter_fpr = filter_fp / neg_x.size(0)
    tp_filter_fpr = tp_filter_fp / neg_x.size(0)
    fn_filter_fpr = fn_filter_fp / neg_x.size(0)
    ne_filter_fpr = ne_filter_fp / neg_x.size(0)

    print(f"{filter_fpr=}")
    print(f"{tp_filter_fpr=}")
    print(f"{fn_filter_fpr=}")
    print(f"{ne_filter_fpr=}")

    print(f"{filter.size_in_bytes()=}")
    print(f"{filter.tp_filter.size_in_bytes()=}")
    print(f"{filter.fn_filter.size_in_bytes()=}")
    print(f"{ne_filter.size_in_bytes()=}")


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
