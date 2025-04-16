import random
import torch
import math
import numpy as np
from model import URLClassifer
from data import LabeledURLDataset, FilteredURLDataset
from filters import DowntownBodegaFilter, NaorEylonFilter, HashSetFilter
from train import find_threshold, test, train, attack
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar


def main(cfg: dict) -> None:
    # Seed.
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    # Load model.
    model = torch.load(cfg["filter"]["model_path"], weights_only=False)
    model = model.cuda()
    torch.compile(model, mode="max-autotune")

    # Load dataset.
    dataset = LabeledURLDataset.from_csv(
        csv_path=cfg["data_path"], seq_len=cfg["seq_len"]
    )
    items = dataset.X[dataset.y > 0.99]
    not_items = dataset.X[dataset.y < 0.01]

    # Find optimal threshold.
    threshold = find_threshold(
        model=model,
        dataset=dataset,
        batch_size=cfg["batch_size"],
        filter_fpr=cfg["filter"]["fpr"],
        max_filter_fpr=cfg["filter"]["max_fpr"],
        verbose=True,
    )
    model_tpr, _, model_tnr, _ = test(
        model=model,
        dataset=dataset,
        batch_size=cfg["batch_size"],
        threshold=threshold,
        num_workers=cfg["num_workers"],
        verbose=True,
    )
    print()

    # Build Downtown Dodega filter.
    db_filter = DowntownBodegaFilter(
        items=items,
        filter_fpr=cfg["filter"]["fpr"],
        model_tpr=model_tpr,
        model_tnr=model_tnr,
        model=model,
        threshold=threshold,
        batch_size=cfg["batch_size"],
        fn_filter_key=cfg["filter"]["fn_filter_key"],
        tp_filter_key=cfg["filter"]["tp_filter_key"],
    )
    assert db_filter.batched_contains(items).sum().item() == items.size(0)
    db_emprical_fpr = db_filter.batched_contains(
        not_items
    ).sum().item() / not_items.size(0)
    print(f"DBF: requested expected fpr = {100 * cfg['filter']['fpr']:.2f}%")
    print(f"DBF: theoretical expected fpr = {100 * db_filter.fpr:.2f}%")
    print(f"DBF: emprical expected fpr = {100 * db_emprical_fpr:.2f}%")
    fn_filter_fpr = db_filter.fn_filter.fpr
    tp_filter_fpr = db_filter.tp_filter.fpr
    print(f"DBF: requested max fpr = {100 * cfg['filter']['max_fpr']:.2f}%")
    print(
        f"DBF: theoretical max fpr = max({100 * fn_filter_fpr:.2f}%, {100 * tp_filter_fpr:.2f}%) = {100 * max(fn_filter_fpr, tp_filter_fpr):.2f}%"
    )
    print(f"DBF: size = {db_filter.size_in_bytes() / 1_000_000:.4f}MB")
    print()

    # Build Naor-Eylon filter.
    fpr = math.exp((-8 * db_filter.size_in_bytes() * math.log(2) ** 2) / items.size(0))
    ne_filter = NaorEylonFilter(
        items=items, fpr=fpr, key=cfg["filter"]["ne_filter_key"]
    )
    assert ne_filter.batched_contains(items).sum().item() == items.size(0)
    ne_emprical_fpr = ne_filter.batched_contains(
        not_items
    ).sum().item() / not_items.size(0)
    print(f"NEF: requested max fpr = {100 * cfg['filter']['fpr']:.2f}%")
    print(f"NEF: theoretical max fpr = {100 * ne_filter.fpr:.2f}%")
    print(f"NEF: emprical max fpr = {100 * ne_emprical_fpr:.2f}%")
    print(f"NEF: size = {ne_filter.size_in_bytes() / 1_000_000:.4f}MB")
    print()

    # Build Downtown Bodega filtered dataset.
    db_dataset = FilteredURLDataset(
        filter=ne_filter, count=cfg["synthetic"]["samples"], seq_len=cfg["seq_len"]
    )

    # Create synthetic Downtown Bodega model.
    db_model = URLClassifer(
        hidden_dim=cfg["synthetic"]["hidden_dim"],
        num_heads=cfg["synthetic"]["num_heads"],
        num_layers=cfg["synthetic"]["num_layers"],
        seq_len=cfg["seq_len"],
        dropout=cfg["synthetic"]["dropout"],
    )
    if not Path(cfg["synthetic"]["db_model_path"]).exists():
        used_items = HashSetFilter(dataset.X)

        def on_epoch_end() -> None:
            used_items.batched_add(db_dataset.X)
            db_dataset.reset()

        db_model = db_model.cuda()
        torch.compile(db_model, mode="max-autotune")
        train(
            db_model,
            db_dataset,
            epochs=cfg["synthetic"]["epochs"],
            batch_size=cfg["batch_size"],
            learning_rate=cfg["synthetic"]["learning_rate"],
            threshold=cfg["synthetic"]["threshold"],
            num_workers=cfg["num_workers"],
            verbose=True,
            on_epoch_end=on_epoch_end,
        )
        torch.save(db_model.state_dict(), cfg["synthetic"]["db_model_path"])
        torch.save(used_items, cfg["synthetic"]["used_items_path"])
    else:
        db_model.load_state_dict(torch.load(cfg["synthetic"]["db_model_path"]))
        db_model = db_model.cuda()
        torch.compile(db_model, mode="max-autotune")
        used_items = torch.load(cfg["synthetic"]["used_items_path"], weights_only=False)

    # Attack the Downtown Bodega filter.
    # Probability of duplicates/true positives is basically zero.
    rnd_items = torch.randint(0, 256, (cfg["attack"]["samples"], cfg["seq_len"]))
    adv_items = attack(
        model=db_model,
        items=rnd_items,
        epochs=cfg["attack"]["epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["attack"]["learning_rate"],
    )
    # Have to be careful here. adv_items might contain true positives or elements
    # we have already queried. We need to filter these out.
    not_used_mask = ~used_items.batched_contains(adv_items)
    rnd_items = rnd_items[not_used_mask]
    adv_items = adv_items[not_used_mask]
    rnd_preds = db_filter.batched_contains(rnd_items)
    adv_preds = db_filter.batched_contains(adv_items)
    print(f"DBF: emprical rnd_fpr = {rnd_preds.float().mean()}")
    print(f"DBF: emprical adv_fpr = {adv_preds.float().mean()}")
    rnd_preds_p = rnd_preds > 0.99
    rnd_preds_n = rnd_preds < 0.01
    adv_preds_p = adv_preds > 0.99
    adv_preds_n = adv_preds < 0.01
    db_table = np.array(
        [
            [
                torch.sum(rnd_preds_p & adv_preds_p).item(),
                torch.sum(rnd_preds_p & adv_preds_n).item(),
            ],
            [
                torch.sum(rnd_preds_n & adv_preds_p).item(),
                torch.sum(rnd_preds_n & adv_preds_n).item(),
            ],
        ]
    )
    print("DBF: contingency_table = ")
    print(db_table)
    db_mcnemar_stat = mcnemar(db_table)
    print("DBF: mcnemar_stat = ")
    print(db_mcnemar_stat)

    hints = db_filter.batched_hints(rnd_items).cpu()
    db_emprical_max_fpr = max(
        db_filter.batched_contains(rnd_items[hints]).float().mean().item(),
        db_filter.batched_contains(rnd_items[hints]).float().mean().item(),
    )
    print(f"DBF: emprical max fpr = {100 * db_emprical_max_fpr:.2f}")


if __name__ == "__main__":
    cfg = {
        # Seed.
        "seed": 42,
        # Dataset.
        "data_path": "data/data.csv",
        "num_workers": 8,
        "batch_size": 1024,
        "seq_len": 128,
        # DB filter model.
        "filter": {
            "fpr": 0.05,
            "max_fpr": 0.2,
            "model_path": "models/db_filter_model.pt",
            "fn_filter_key": b'B\xae\x1a\xc2\xbf\x8e\xb3\x94w\xdab\x8b"}/\xb5\x93Hrg \xedY\x9b5\xce\x85\xc7u#7\xb3',
            "tp_filter_key": b"JS\xf8\xed!\xc4\x84\xc7z\x0f\xad+k\x94\xe3E\xe8xG\xea\x1bRu\xa7\xe0\xd2d\xf8C\xf3\x0c\xad",
            "ne_filter_key": b"\xd04\xad\xaa\xd9%\x83s\xde\xa4\xab\xb4\xc8\x8d|w\xd9\xcf\x88z\x8bk\xf9\xdf\xb6\xc99\xacR\x88\x08F",
        },
        # DB filter synthetic model.
        "synthetic": {
            # Training.
            "epochs": 10,
            "samples": 256 * 1024,
            "learning_rate": 0.0001,
            "threshold": 0.5,
            # Model.
            "hidden_dim": 128,
            "num_heads": 8,
            "num_layers": 4,
            "dropout": 0.1,
            "db_model_path": "models/synthetic_db_filter.pt",
            "used_items_path": "models/synthetic_db_filter_used_items.pt",
        },
        # Attack.
        "attack": {
            "epochs": 30,
            "samples": 16 * 1024,
            "learning_rate": 0.1,
        },
    }
    main(cfg)
