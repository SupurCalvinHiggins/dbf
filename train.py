import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from data import URLDataset
from model import URLClassifer


def train(model: URLClassifer, dataset: URLDataset, cfg: dict) -> URLClassifer:
    loader = DataLoader(
        dataset, cfg["batch_size"], shuffle=True, pin_memory=True, num_workers=4
    )

    model = model.to(device=cfg["device"])
    criteron = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda")

    for _ in range(cfg["epochs"]):
        for _, x, y in loader:
            x, y = (
                x.to(cfg["device"], non_blocking=True),
                y.to(cfg["device"], non_blocking=True),
            )

            with torch.autocast("cuda", torch.float16):
                pred_y = model(x)
                loss = criteron(pred_y, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    return model


def main():
    model = URLClassifer(128, 4, 2)
    dataset = URLDataset.from_csv(Path("data/malicious_phish.csv"))
    cfg = {
        "batch_size": 32,
        "lr": 0.0001,
        "device": torch.device("cuda"),
        "epochs": 10,
    }
    train(model, dataset, cfg)


if __name__ == "__main__":
    main()
