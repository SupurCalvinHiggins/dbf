import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from data import URLDataset
from model import URLClassifer
from tqdm import tqdm


device = torch.device("cuda")


def summary(model: URLClassifer) -> None:
    total_size_bytes = 0
    for param in model.parameters():
        total_size_bytes += param.numel() * param.element_size()
    print(f"Model [1/1]: size = {total_size_bytes}B")
    print(f"Model [1/1]: size = {total_size_bytes / 1_000_000:.4f}MB")


@torch.enable_grad()
def train(
    model: URLClassifer,
    dataset: URLDataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    threshold: float,
    num_workers: int,
) -> URLClassifer:
    model.train()

    loader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    criteron = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, fused=True)
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        tp = torch.tensor(0.0, device=device)
        fp = torch.tensor(0.0, device=device)
        tn = torch.tensor(0.0, device=device)
        fn = torch.tensor(0.0, device=device)

        for _, x, y in tqdm(loader, desc=f"Epoch [{epoch + 1}/{epochs}]"):
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

        tp = tp.item()
        fp = fp.item()
        tn = tn.item()
        fn = fn.item()

        count = tp + fp + tn + fn
        tpr = tp / count
        fpr = fp / count
        tnr = tn / count
        fnr = fn / count
        acc = tpr + tnr

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
) -> None:
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

    for _, x, y in tqdm(loader, desc="Test [1/1]"):
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

    count = tp + fp + tn + fn
    tpr = tp / count
    fpr = fp / count
    tnr = tn / count
    fnr = fn / count
    acc = tpr + tnr

    print(f"Test [1/1]: acc = {100 * acc:.2f}%")
    print(f"Test [1/1]: tpr = {100 * tpr:.2f}%")
    print(f"Test [1/1]: fpr = {100 * fpr:.2f}%")
    print(f"Test [1/1]: tnr = {100 * tnr:.2f}%")
    print(f"Test [1/1]: fnr = {100 * fnr:.2f}%")


def main(cfg: dict) -> None:
    model = URLClassifer(
        cfg["hidden_dim"],
        cfg["num_heads"],
        cfg["num_layers"],
        dropout=cfg["dropout"],
        seq_len=cfg["seq_len"],
    ).to(device=device)
    torch.compile(model, mode="max-autotune")
    summary(model)

    train_dataset = URLDataset.from_csv(Path(cfg["train_path"]), seq_len=cfg["seq_len"])
    model = train(
        model,
        train_dataset,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        threshold=cfg["threshold"],
        num_workers=cfg["num_workers"],
    )

    test_dataset = URLDataset.from_csv(Path(cfg["test_path"]), seq_len=cfg["seq_len"])
    test(
        model,
        test_dataset,
        batch_size=cfg["batch_size"],
        threshold=cfg["threshold"],
        num_workers=cfg["num_workers"],
    )

    model = model.cpu()
    torch.save(model.state_dict(), cfg["model_path"])


if __name__ == "__main__":
    cfg = {
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
        "model_path": "models/transformer.pt",
        # Training.
        "epochs": 1,
        "batch_size": 1024,
        "learning_rate": 0.0001,
    }
    main(cfg)
