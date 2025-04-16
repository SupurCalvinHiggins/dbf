import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from pathlib import Path
from torch import Tensor
from torch.utils.data import DataLoader
from data import URLDataset, LabeledURLDataset
from model import URLClassifer
from tqdm import tqdm
from typing import Optional, Callable


device = torch.device("cuda")


@torch.enable_grad()
def train(
    model: URLClassifer,
    dataset: URLDataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    threshold: float,
    num_workers: int,
    verbose: bool = False,
    on_epoch_end: Optional[Callable[[], None]] = None,
) -> URLClassifer:
    model.train()

    loader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    pr = dataset.y.sum() / dataset.y.size(0)
    pos_weight = torch.tensor([(1 - pr) / pr], device=device)
    criteron = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, fused=True)
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        tp = torch.tensor(0.0, device=device)
        fp = torch.tensor(0.0, device=device)
        tn = torch.tensor(0.0, device=device)
        fn = torch.tensor(0.0, device=device)

        for x, y in tqdm(
            loader, desc=f"Epoch [{epoch + 1}/{epochs}]", disable=not verbose
        ):
            x, y = (
                x.to(device, non_blocking=True),
                y.to(device, non_blocking=True),
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", torch.float16):
                pred_y = model(x)
                loss = criteron(pred_y, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            p = y > threshold
            n = ~p

            pred_p = F.sigmoid(pred_y) > threshold
            pred_n = ~pred_p

            tp += (p & pred_p).sum()
            fp += (n & pred_p).sum()
            tn += (n & pred_n).sum()
            fn += (p & pred_n).sum()

        if on_epoch_end is not None:
            on_epoch_end()

        tp = tp.item()
        fp = fp.item()
        tn = tn.item()
        fn = fn.item()

        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        tnr = tn / (tn + fp)
        fnr = fn / (tp + fn)
        acc = (tn + tp) / (tp + fp + tn + fn)

        if verbose:
            print(f"Epoch [{epoch + 1}/{epochs}]: acc = {100 * acc:.2f}%")
            print(f"Epoch [{epoch + 1}/{epochs}]: tpr = {100 * tpr:.2f}%")
            print(f"Epoch [{epoch + 1}/{epochs}]: fpr = {100 * fpr:.2f}%")
            print(f"Epoch [{epoch + 1}/{epochs}]: tnr = {100 * tnr:.2f}%")
            print(f"Epoch [{epoch + 1}/{epochs}]: fnr = {100 * fnr:.2f}%")

    return model


@torch.no_grad()
def test(
    model: URLClassifer,
    dataset: URLDataset,
    batch_size: int,
    threshold: float,
    num_workers: int,
    verbose: bool = False,
) -> tuple[float, float, float, float]:
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
    )

    tp = torch.tensor(0.0, device=device)
    fp = torch.tensor(0.0, device=device)
    tn = torch.tensor(0.0, device=device)
    fn = torch.tensor(0.0, device=device)

    for x, y in tqdm(loader, desc="Test [1/1]", disable=not verbose):
        x, y = (
            x.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
        )

        with torch.autocast("cuda", torch.float16):
            pred_y = model(x)

        p = y > threshold
        n = ~p

        pred_p = F.sigmoid(pred_y) > threshold
        pred_n = ~pred_p

        tp += (p & pred_p).sum()
        fp += (n & pred_p).sum()
        tn += (n & pred_n).sum()
        fn += (p & pred_n).sum()

    tp = tp.item()
    fp = fp.item()
    tn = tn.item()
    fn = fn.item()

    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    tnr = tn / (tn + fp)
    fnr = fn / (tp + fn)
    acc = (tn + tp) / (tp + fp + tn + fn)

    if verbose:
        print(f"Test [1/1]: acc = {100 * acc:.2f}%")
        print(f"Test [1/1]: tpr = {100 * tpr:.2f}%")
        print(f"Test [1/1]: fpr = {100 * fpr:.2f}%")
        print(f"Test [1/1]: tnr = {100 * tnr:.2f}%")
        print(f"Test [1/1]: fnr = {100 * fnr:.2f}%")

    return tpr, fpr, tnr, fnr


def predict(
    model: URLClassifer,
    items: Tensor,
    batch_size: int,
    verbose: bool = False,
) -> Tensor:
    model.eval()

    result = torch.zeros(items.size(0), device=device, dtype=torch.float)

    for i in tqdm(
        range(0, items.size(0), batch_size), desc="Predict [1/1]", disable=not verbose
    ):
        x = items[i : i + batch_size].to(device, non_blocking=True)

        with torch.autocast("cuda", torch.float16):
            pred_y = F.sigmoid(model(x))

        result[i : i + batch_size] = pred_y

    return result


@torch.no_grad()
def find_threshold(
    model: URLClassifer,
    dataset: URLDataset,
    batch_size: int,
    filter_fpr: float,
    max_filter_fpr: float,
    verbose: bool = False,
) -> float:
    y = dataset.y
    pred_y = predict(model, dataset.X, batch_size, verbose=verbose).cpu()

    idx = torch.argsort(pred_y)
    y = y[idx]
    pred_y = pred_y[idx]

    p_mask = y > 0.99
    n_mask = y < 0.01

    # n_count[i] is the number of negative examples in y[:i+1]
    n_count = torch.cumsum(n_mask, dim=-1)

    # p_count[i] is the number of postive examples in y[i+1:]
    p_count = torch.sum(p_mask) - torch.cumsum(p_mask, dim=-1)

    # TODO: vectorize.
    n = len(y)
    best_cost, best_i = float("inf"), 0
    for i in tqdm(range(1, n - 1), disable=not verbose):
        tpr = p_count[i - 1] / p_count[0]
        fnr = 1 - tpr
        tnr = n_count[i - 1] / n_count[-1]
        fpr = 1 - tnr

        if (
            math.isclose(tpr, 0.0)
            or math.isclose(fnr, 0.0)
            or math.isclose(tnr, 0.0)
            or math.isclose(fpr, 0.0)
        ):
            continue

        cost = -tpr * math.log(tpr / fpr) - fnr * math.log(fnr / tnr)
        max_fpr = max((tpr / fpr).item(), (fnr / tnr).item()) * filter_fpr
        if cost < best_cost and max_fpr < max_filter_fpr:
            best_cost = cost
            best_i = i

    assert best_cost < float("inf")

    threshold = ((pred_y[best_i] + pred_y[best_i + 1]) / 2).item()

    if verbose:
        print(f"Find [1/1]: best_cost = {best_cost}")
        print(f"Find [1/1]: threshold = {threshold}")

    return threshold


@torch.enable_grad()
def attack(
    model: URLClassifer,
    items: Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    verbose: bool = True,
) -> Tensor:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    x = nn.Parameter(model.embedding(items.clone().to(device)))
    y = torch.ones((items.size(0),), dtype=torch.float, device=device)

    criteron = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam([x], lr=learning_rate, fused=True)

    for epoch in range(epochs):
        total_loss = torch.tensor(0.0, device=device)

        for i in tqdm(
            range(0, len(x), batch_size),
            desc=f"Attack [{epoch + 1}/{epochs}]",
            disable=not verbose,
        ):
            optimizer.zero_grad(set_to_none=True)

            pred_y = model(x[i : i + batch_size], embed=False)
            loss = criteron(pred_y, y[i : i + batch_size])
            loss.backward()

            optimizer.step()

            total_loss += loss

        if verbose:
            print(f"Attack [{epoch + 1}/{epochs}] loss = {total_loss.item()}")

    normed_x = F.normalize(x, p=2, dim=-1)
    normed_w = F.normalize(model.embedding.weight, p=2, dim=-1)
    cos_sim = normed_x @ normed_w.T
    x = torch.argmax(cos_sim, dim=-1)

    return x


def main(cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    model = URLClassifer(
        cfg["hidden_dim"],
        cfg["num_heads"],
        cfg["num_layers"],
        dropout=cfg["dropout"],
        seq_len=cfg["seq_len"],
    ).to(device=device)
    torch.compile(model, mode="max-autotune")
    print(f"Model [1/1]: size = {model.size_in_bytes()}B")
    print(f"Model [1/1]: size = {model.size_in_bytes() / 1_000_000:.4f}MB")

    train_dataset = LabeledURLDataset.from_csv(
        Path(cfg["train_path"]), seq_len=cfg["seq_len"]
    )
    model = train(
        model,
        train_dataset,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        threshold=cfg["threshold"],
        num_workers=cfg["num_workers"],
        verbose=True,
    )

    test_dataset = LabeledURLDataset.from_csv(
        Path(cfg["test_path"]), seq_len=cfg["seq_len"]
    )
    test(
        model,
        test_dataset,
        batch_size=cfg["batch_size"],
        threshold=cfg["threshold"],
        num_workers=cfg["num_workers"],
        verbose=True,
    )

    model = model.cpu()
    torch.save(model, cfg["model_path"])


if __name__ == "__main__":
    cfg = {
        # Seed.
        "seed": 42,
        # Dataset.
        "train_path": "data/train.csv",
        "test_path": "data/test.csv",
        "seq_len": 128,
        "num_workers": 8,
        # Model.
        "hidden_dim": 8,
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.1,
        "threshold": 0.5,
        "model_path": "models/db_filter_model.pt",
        # Training.
        "epochs": 10,
        "batch_size": 1024,
        "learning_rate": 0.0001,
    }
    main(cfg)
