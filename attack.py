# TODO: random URL dataset with labels from model
# TODO: train model to replicate filter using train.py

import torch
import math
from model import URLClassifer
from data import LabeledURLDataset, FilteredURLDataset
from filters import DowntownBodegaFilter, NaorEylonFilter
from train import find_threshold, test


def main(cfg: dict) -> None:
    # Load model.
    model = URLClassifer(
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        seq_len=cfg["seq_len"],
        dropout=0.0,
    )
    model.load_state_dict(torch.load(cfg["model_path"]))
    model = model.cuda()
    torch.compile(model, mode="max-autotune")

    # Load dataset.
    dataset = LabeledURLDataset.from_csv(
        csv_path=cfg["test_path"], seq_len=cfg["seq_len"]
    )
    items = dataset.X[dataset.y > 0.99]
    not_items = dataset.X[dataset.y < 0.01]

    # Find optimal threshold.
    threshold = find_threshold(
        model=model,
        dataset=dataset,
        fpr=cfg["fpr"],
        batch_size=cfg["batch_size"],
        verbose=True,
    )
    tpr, fpr, _, _ = test(
        model=model,
        dataset=dataset,
        batch_size=cfg["batch_size"],
        threshold=threshold,
        num_workers=8,
        verbose=True,
    )
    print()

    # Build Downtown Dodega filter.
    db_filter = DowntownBodegaFilter(
        items=items,
        fpr=fpr,
        tpr=tpr,
        model=model,
        threshold=threshold,
        batch_size=cfg["batch_size"],
        fn_filter_key=cfg["fn_filter_key"],
        tp_filter_key=cfg["tp_filter_key"],
    )
    assert db_filter.batched_contains(items).sum().item() == items.size(0)
    db_actual_fpr = db_filter.batched_contains(not_items).sum().item() / not_items.size(
        0
    )
    print(f"DBF: actual fpr = {100 * db_actual_fpr:.2f}%")
    print(f"DBF: expected fpr = {100 * cfg['fpr']:.2f}%")
    print(f"DBF: size = {db_filter.size_in_bytes() / 1_000_000:.4f}MB")
    print()

    # Build Naor-Eylon filter.
    fpr = math.exp((-8 * db_filter.size_in_bytes() * math.log(2) ** 2) / items.size(0))
    ne_filter = NaorEylonFilter(items=items, fpr=fpr, key=cfg["fn_filter_key"])
    assert ne_filter.batched_contains(items).sum().item() == items.size(0)
    ne_actual_fpr = ne_filter.batched_contains(not_items).sum().item() / not_items.size(
        0
    )
    print(f"NEF: actual_fpr = {100 * ne_actual_fpr:.2f}%")
    print(f"NEF: expected_fpr = {100 * cfg['fpr']:.2f}%")
    print(f"NEF: size = {ne_filter.size_in_bytes() / 1_000_000:.4f}MB")
    print()


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
        # Filters.
        "fpr": 0.05,
        "fn_filter_key": b'B\xae\x1a\xc2\xbf\x8e\xb3\x94w\xdab\x8b"}/\xb5\x93Hrg \xedY\x9b5\xce\x85\xc7u#7\xb3',
        "tp_filter_key": b"JS\xf8\xed!\xc4\x84\xc7z\x0f\xad+k\x94\xe3E\xe8xG\xea\x1bRu\xa7\xe0\xd2d\xf8C\xf3\x0c\xad",
    }
    main(cfg)
