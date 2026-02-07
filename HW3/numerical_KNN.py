import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


class KNNNumeric:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        return self

    def predict_one(self, x: np.ndarray):
        dists = np.array([euclidean_distance(x, xi) for xi in self.X_train])
        nn_idx = np.argsort(dists)[: self.k]
        nn_labels = self.y_train[nn_idx]

        
        values, counts = np.unique(nn_labels, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X: np.ndarray):
        return np.array([self.predict_one(x) for x in X])


data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


Ks = list(range(1, 21))
acc_manual = []
acc_sklearn = []

for k in Ks:
    # manual KNN
    knn_m = KNNNumeric(k=k).fit(X_train, y_train)
    y_pred_m = knn_m.predict(X_test)
    acc_manual.append(accuracy_score(y_test, y_pred_m))

    # sklearn KNN
    knn_s = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn_s.fit(X_train, y_train)
    y_pred_s = knn_s.predict(X_test)
    acc_sklearn.append(accuracy_score(y_test, y_pred_s))


eps = 0.0005  # very small visual offset

plt.figure()
plt.plot(Ks, acc_manual, marker="o", label="Manual KNN (Euclidean)")
plt.plot(Ks, [a + eps for a in acc_sklearn],
         marker="s", linestyle="--",
         label="sklearn KNN (Euclidean)")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Part 1: Breast Cancer | Accuracy vs K")
plt.grid(True)
plt.legend()
plt.show()


print("Best Manual:", max(acc_manual), "at K =", Ks[int(np.argmax(acc_manual))])
print("Best sklearn:", max(acc_sklearn), "at K =", Ks[int(np.argmax(acc_sklearn))])
