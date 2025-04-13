from rbloom import Bloom
from blake3 import blake3
from torch import Tensor
import torch

n = 112794
fpr = 0.05
filter = Bloom(n, fpr)
print(filter.size_in_bits)


class NaorEylonFilter:
    def __init__(self, n, fpr: float) -> None:
        self.filter = Bloom(n, fpr, hash_func=self.hash)
        self.key = b"JS\xf8\xed!\xc4\x84\xc7z\x0f\xad+k\x94\xe3E\xe8xG\xea\x1bRu\xa7\xe0\xd2d\xf8C\xf3\x0c\xad"
        self.fpr = fpr

        for _ in range(n):
            self.filter.add(torch.randint(0, 255, (128,)))

    def hash(self, item: Tensor) -> int:
        data = item.cpu().numpy().tobytes()
        return int.from_bytes(blake3(data, key=self.key).digest()[:16], signed=True)

    def batched_contains(self, items: Tensor) -> Tensor:
        return torch.tensor([item in self.filter for item in items], dtype=torch.bool)

    def size_in_bytes(self) -> int:
        return self.filter.size_in_bits // 8


filter = NaorEylonFilter(n, fpr)
print(filter.filter.size_in_bits)
