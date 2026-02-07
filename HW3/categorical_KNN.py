import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
local_path = Path(__file__).parent / "car.data"
df = pd.read_csv(local_path, header=None, names=cols)


X = df[cols[:-1]].astype(str).values  # categorical strings
y = df["class"].astype(str).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


class KNNCategorical:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        return self

    def predict_one(self, x: np.ndarray):
        dists = np.array([hamming_distance(x, xi) for xi in self.X_train])
        nn_idx = np.argsort(dists)[: self.k]
        nn_labels = self.y_train[nn_idx]

        values, counts = np.unique(nn_labels, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X: np.ndarray):
        return np.array([self.predict_one(x) for x in X])


def ordinal_encode_train_test(X_train, X_test):
    Xtr = X_train.copy()
    Xte = X_test.copy()
    Xtr_enc = np.zeros_like(Xtr, dtype=int)
    Xte_enc = np.zeros_like(Xte, dtype=int)

    for j in range(Xtr.shape[1]):
        cats = sorted(set(Xtr[:, j].tolist()))
        mapping = {c: i for i, c in enumerate(cats)}
        # encode train
        Xtr_enc[:, j] = [mapping[v] for v in Xtr[:, j]]
        # encode test (if unseen category appears, push to -1)
        Xte_enc[:, j] = [mapping.get(v, -1) for v in Xte[:, j]]

    return Xtr_enc, Xte_enc

X_train_enc, X_test_enc = ordinal_encode_train_test(X_train, X_test)


Ks = list(range(1, 21))
acc_manual = []
acc_sklearn = []

for k in Ks:
    # manual
    knn_m = KNNCategorical(k=k).fit(X_train, y_train)
    y_pred_m = knn_m.predict(X_test)
    acc_manual.append(accuracy_score(y_test, y_pred_m))

    # sklearn (hamming)
    knn_s = KNeighborsClassifier(n_neighbors=k, metric="hamming")
    knn_s.fit(X_train_enc, y_train)
    y_pred_s = knn_s.predict(X_test_enc)
    acc_sklearn.append(accuracy_score(y_test, y_pred_s))


plt.figure()
plt.plot(Ks, acc_manual, marker="o", label="Manual KNN (Hamming)")
plt.plot(Ks, acc_sklearn, marker="s", label='sklearn KNN (metric="hamming")')
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Part 2: Car Evaluation | Accuracy vs K")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("Best Manual:", max(acc_manual), "at K =", Ks[int(np.argmax(acc_manual))])
print("Best sklearn:", max(acc_sklearn), "at K =", Ks[int(np.argmax(acc_sklearn))])
