from __future__ import annotations

import torch
import csv
from torch.utils.data import Dataset
from torch import Tensor
from pathlib import Path


class URLDataset(Dataset):
    @staticmethod
    def from_csv(csv_path: Path, seq_len: int = 128) -> URLDataset:
        urls = []
        labels = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for url, label in reader:
                urls.append(url)
                labels.append(label != "benign")

        return URLDataset(urls, labels, seq_len)

    def __init__(self, urls: list[str], labels: list[bool], seq_len: int) -> None:
        clean_urls = [url.encode().ljust(seq_len, b"\x00")[:seq_len] for url in urls]
        self.X = torch.tensor([list(url) for url in clean_urls], dtype=torch.int)
        self.y = torch.tensor(labels, dtype=torch.float)

    def __getitem__(self, index: int) -> tuple[int, Tensor, Tensor]:
        return index, self.X[index], self.y[index]

    def __len__(self) -> int:
        return len(self.y)
