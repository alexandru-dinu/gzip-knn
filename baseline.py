import multiprocessing as mp
import zlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tqdm import tqdm


def com(xs: str) -> bytes:
    return zlib.compress(bytes(xs.encode()))


def dec(zs: bytes) -> str:
    return zlib.decompress(zs).decode()


def ncd(x: str, y: str) -> float:
    cx, cy = len(com(x)), len(com(y))
    cxy = len(com(f"{x}{y}"))
    return (cxy - min(cx, cy)) / max(cx, cy)


class KNN:
    def __init__(self, kb_x, kb_y):
        assert len(kb_x) == len(kb_y)
        self.kb_x = kb_x
        self.kb_y = kb_y

    def __len__(self):
        return len(self.kb_x)

    def classify(self, query: str, k: int) -> int:
        labels = [0] * 2  # TODO

        ix = range(len(self))
        for l in sorted(ix, key=lambda i: ncd(self.kb_x[i], query))[:k]:
            labels[self.kb_y[l]] += 1

        pred = 0
        for i in range(1, len(labels)):
            if labels[i] > labels[pred]:
                pred = i

        return pred


def _work(i):
    p = knn.classify(X_test[i], k=50)
    t = y_test[i]

    return p, t


if __name__ == "__main__":
    df = pd.read_csv(
        "./data/quora-insincere/train.csv", usecols=["question_text", "target"]
    ).query("question_text.str.len() >= 100")
    print(df.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        df["question_text"].tolist(),
        df["target"].tolist(),
        test_size=1 / 3,
        shuffle=True,
        random_state=42,
    )
    print(f"{len(X_train)=}")
    print(f"{len(X_test)=}")

    knn = KNN(X_train, y_train)

    n = 100
    ps = []
    ts = []
    with tqdm(total=n) as pbar:
        with mp.Pool(16) as pool:
            for (p, t) in pool.imap_unordered(_work, range(n)):
                ps.append(p)
                ts.append(t)
                pbar.update(1)
                pbar.set_description(f"F1: {metrics.f1_score(ts, ps)}")
