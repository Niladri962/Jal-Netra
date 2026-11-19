"""Isolation Forest in pure NumPy (Liu, Ting & Zhou, 2008).

Small, dependency-free, and vectorised across all trees at once so scoring a
batch of nodes every 15 minutes is cheap. Keeping it in-house means the
serverless deployment does not need scikit-learn / SciPy.

``decision_function`` follows scikit-learn's convention (contamination="auto"):
    decision = 0.5 - anomaly_score      (< 0 => more anomalous than average)
"""
from __future__ import annotations

import math

import numpy as np

EULER = 0.5772156649


def _c(n: float) -> float:
    """Average path length of an unsuccessful BST search over n points."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + EULER) - 2.0 * (n - 1) / n


class IsolationForest:
    def __init__(self, n_estimators: int = 150, max_samples: int = 256, random_state: int = 0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray) -> "IsolationForest":
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        psi = min(self.max_samples, n)
        self.max_depth = int(math.ceil(math.log2(max(psi, 2))))
        self.c_psi = _c(psi)
        max_nodes = 2 ** (self.max_depth + 1)
        T = self.n_estimators
        self.feat = np.full((T, max_nodes), -1, dtype=np.int64)
        self.thr = np.zeros((T, max_nodes))
        self.left = np.zeros((T, max_nodes), dtype=np.int64)
        self.right = np.zeros((T, max_nodes), dtype=np.int64)
        self.leaf_val = np.zeros((T, max_nodes))      # depth + c(size) at leaves
        for t in range(T):
            idx = self.rng.choice(n, psi, replace=False)
            self._grow(t, X[idx], d)
        return self

    def _grow(self, t: int, Xs: np.ndarray, d: int) -> None:
        stack = [(0, Xs, 0)]
        nxt = 1
        while stack:
            node, data, depth = stack.pop()
            m = len(data)
            if depth >= self.max_depth or m <= 1:
                self.leaf_val[t, node] = depth + _c(m)
                continue
            lo, hi = data.min(axis=0), data.max(axis=0)
            usable = np.flatnonzero(hi > lo)
            if usable.size == 0:
                self.leaf_val[t, node] = depth + _c(m)
                continue
            f = int(self.rng.choice(usable))
            s = self.rng.uniform(lo[f], hi[f])
            mask = data[:, f] < s
            l, r = nxt, nxt + 1
            nxt += 2
            self.feat[t, node], self.thr[t, node] = f, s
            self.left[t, node], self.right[t, node] = l, r
            stack.append((l, data[mask], depth + 1))
            stack.append((r, data[~mask], depth + 1))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score s in (0, 1]; ~0.5 normal, -> 1 anomalous."""
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        T = self.n_estimators
        node = np.zeros((T, n), dtype=np.int64)
        trees = np.arange(T)[:, None]
        rows = np.arange(n)[None, :]
        for _ in range(self.max_depth):
            f = self.feat[trees, node]
            internal = f >= 0
            if not internal.any():
                break
            xv = X[rows, np.where(internal, f, 0)]
            go_left = xv < self.thr[trees, node]
            child = np.where(go_left, self.left[trees, node], self.right[trees, node])
            node = np.where(internal, child, node)
        path = self.leaf_val[trees, node].mean(axis=0)
        return 2.0 ** (-path / self.c_psi)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return 0.5 - self.score_samples(X)
