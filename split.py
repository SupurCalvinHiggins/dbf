import csv
import random


def main(
    data_path: str, train_path: str, test_path: str, train_frac: float, seed: int = 42
) -> None:
    random.seed(seed)

    with open(data_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    random.shuffle(rows)
    n = len(rows)

    sp = int(n * train_frac)
    train_rows = rows[:sp]
    test_rows = rows[sp:]

    with open(train_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in train_rows:
            writer.writerow(row)

    with open(test_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in test_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main("data/data.csv", "data/train.csv", "data/test.csv", 0.8)
