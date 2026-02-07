import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

local_path = Path(__file__).parent / "tips.csv"
df = pd.read_csv(local_path)


y = df["time"].astype(str).values


cat_cols = ["sex", "smoker", "day"]
num_cols = ["total_bill", "tip", "size"]

X_cat = df[cat_cols].astype(str).values
X_num = df[num_cols].astype(float).values


Xc_train, Xc_test, Xn_train, Xn_test, y_train, y_test = train_test_split(
    X_cat, X_num, y, test_size=0.2, random_state=42, stratify=y
)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))

def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))

def mixed_distance(x_num, x_cat, xi_num, xi_cat, W: float) -> float:
    return euclidean_distance(x_num, xi_num) + W * hamming_distance(x_cat, xi_cat)


class KNNMixed:
    def __init__(self, k: int = 5, W: float = 1.0):
        self.k = k
        self.W = W
        self.Xn_train = None
        self.Xc_train = None
        self.y_train = None

    def fit(self, Xn: np.ndarray, Xc: np.ndarray, y: np.ndarray):
        self.Xn_train = Xn
        self.Xc_train = Xc
        self.y_train = y
        return self

    def predict_one(self, x_num: np.ndarray, x_cat: np.ndarray):
        dists = np.array([
            mixed_distance(x_num, x_cat, xi_num, xi_cat, self.W)
            for xi_num, xi_cat in zip(self.Xn_train, self.Xc_train)
        ])
        nn_idx = np.argsort(dists)[: self.k]
        nn_labels = self.y_train[nn_idx]

        values, counts = np.unique(nn_labels, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, Xn: np.ndarray, Xc: np.ndarray):
        return np.array([self.predict_one(xn, xc) for xn, xc in zip(Xn, Xc)])


def encode_categorical_train_test(Xc_train, Xc_test):
    Xtr_enc = np.zeros_like(Xc_train, dtype=int)
    Xte_enc = np.zeros_like(Xc_test, dtype=int)

    for j in range(Xc_train.shape[1]):
        cats = sorted(set(Xc_train[:, j].tolist()))
        mapping = {c: i for i, c in enumerate(cats)}
        Xtr_enc[:, j] = [mapping[v] for v in Xc_train[:, j]]
        Xte_enc[:, j] = [mapping.get(v, -1) for v in Xc_test[:, j]]

    return Xtr_enc, Xte_enc

Xc_train_enc, Xc_test_enc = encode_categorical_train_test(Xc_train, Xc_test)


X_train_sklearn = np.hstack([Xn_train, Xc_train_enc])
X_test_sklearn  = np.hstack([Xn_test,  Xc_test_enc])


Ks = list(range(1, 21))
Ws = [0.5, 1.0, 2.0, 3.0]

acc_manual = {W: [] for W in Ws}
acc_sklearn = {W: [] for W in Ws}

for W in Ws:
    X_train_scaled = np.hstack([Xn_train, (W * Xc_train_enc)])
    X_test_scaled  = np.hstack([Xn_test,  (W * Xc_test_enc)])

    for k in Ks:
        # manual
        knn_m = KNNMixed(k=k, W=W).fit(Xn_train, Xc_train, y_train)
        y_pred_m = knn_m.predict(Xn_test, Xc_test)
        acc_manual[W].append(accuracy_score(y_test, y_pred_m))

        # sklearn baseline
        knn_s = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn_s.fit(X_train_scaled, y_train)
        y_pred_s = knn_s.predict(X_test_scaled)
        acc_sklearn[W].append(accuracy_score(y_test, y_pred_s))


plt.figure()
for W in Ws:
    plt.plot(Ks, acc_manual[W], marker="o", label=f"Manual (W={W})")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Part 3 (Manual): Tips | Accuracy vs K for multiple W")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
for W in Ws:
    plt.plot(Ks, acc_sklearn[W], marker="s", label=f"sklearn baseline (W={W})")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Part 3 (sklearn baseline): Tips | Accuracy vs K for multiple W")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


for W in Ws:
    best_k_m = Ks[int(np.argmax(acc_manual[W]))]
    best_a_m = max(acc_manual[W])
    best_k_s = Ks[int(np.argmax(acc_sklearn[W]))]
    best_a_s = max(acc_sklearn[W])
    print(f"W={W} | Manual best acc={best_a_m:.4f} at K={best_k_m} | sklearn best acc={best_a_s:.4f} at K={best_k_s}")
