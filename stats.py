import csv
import matplotlib.pyplot as plt


def main(data_path: str) -> None:
    with open(data_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    seq_lens = [len(url) for url, _ in rows]
    plt.hist(seq_lens)
    plt.xlabel("Length of URLs")
    plt.show()

    y = [not label.startswith("benign") for _, label in rows]
    false = y.count(False)
    true = y.count(True)
    plt.bar([False, True], [false, true])
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    main("data/data.csv")
