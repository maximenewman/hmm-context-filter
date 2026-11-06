"""Diagonal-Gaussian HMM for streaming keep/drop filtering.

Supervised fit from labeled line sequences (no Baum-Welch needed when we
have labels, which we do for our corpora). Viterbi for offline decode,
forward-filtering for true online/streaming decode.

numpy only. ~150 lines. No hmmlearn, no torch.
"""
import numpy as np


class GaussianHMM:
    def __init__(self, n_states: int, n_features: int):
        self.K = n_states
        self.D = n_features
        self.log_start = np.full(self.K, -np.log(self.K))
        self.log_trans = np.full((self.K, self.K), -np.log(self.K))
        self.mean = np.zeros((self.K, self.D))
        self.var = np.ones((self.K, self.D))

    def fit_supervised(self, seqs, labels, var_floor=1e-3, smooth=1.0):
        """seqs: list of (T, D) arrays. labels: list of (T,) int arrays."""
        K, D = self.K, self.D
        start_c = np.full(K, smooth)
        trans_c = np.full((K, K), smooth)
        acc = [[] for _ in range(K)]
        for X, y in zip(seqs, labels):
            if len(X) == 0:
                continue
            start_c[y[0]] += 1
            for t in range(1, len(y)):
                trans_c[y[t - 1], y[t]] += 1
            for k in range(K):
                rows = X[y == k]
                if len(rows):
                    acc[k].append(rows)
        self.log_start = np.log(start_c / start_c.sum())
        self.log_trans = np.log(trans_c / trans_c.sum(axis=1, keepdims=True))
        for k in range(K):
            if acc[k]:
                M = np.vstack(acc[k])
                self.mean[k] = M.mean(axis=0)
                self.var[k] = np.maximum(M.var(axis=0), var_floor)
        return self

    def _log_emit(self, X):
        """(T, D) -> (T, K) log Gaussian emission."""
        return -0.5 * (
            np.log(2 * np.pi * self.var).sum(axis=1)
            + ((X[:, None, :] - self.mean[None, :, :]) ** 2 / self.var[None, :, :]).sum(axis=2)
        )

    def viterbi(self, X):
        """Most likely state path. Returns (path, logprob)."""
        T = len(X)
        if T == 0:
            return np.array([], dtype=int), 0.0
        E = self._log_emit(X)
        dp = np.full((T, self.K), -np.inf)
        bp = np.zeros((T, self.K), dtype=int)
        dp[0] = self.log_start + E[0]
        for t in range(1, T):
            cand = dp[t - 1][:, None] + self.log_trans
            bp[t] = cand.argmax(axis=0)
            dp[t] = cand.max(axis=0) + E[t]
        path = np.zeros(T, dtype=int)
        path[-1] = dp[-1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = bp[t + 1, path[t + 1]]
        return path, float(dp[-1].max())

    def forward_posteriors(self, X):
        """Online-capable: P(state_t | x_1..t) for each t (filtering, not smoothing)."""
        T = len(X)
        E = self._log_emit(X)
        alpha = np.zeros((T, self.K))
        alpha[0] = self.log_start + E[0]
        alpha[0] -= np.logaddexp.reduce(alpha[0])
        for t in range(1, T):
            m = np.logaddexp.reduce(alpha[t - 1][:, None] + self.log_trans, axis=0)
            alpha[t] = m + E[t]
            alpha[t] -= np.logaddexp.reduce(alpha[t])
        return np.exp(alpha)

    def predict_online(self, X):
        """Streaming decode: argmax of the filtering posterior at each t."""
        return self.forward_posteriors(X).argmax(axis=1)


def state_keep_map(labels_list, decoded_list):
    """Map each state to keep(1)/drop(0) by majority of the gold labels
    falling into it. Gold label convention: 1 = keep, 0 = drop."""
    from collections import defaultdict
    votes = defaultdict(lambda: [0, 0])
    for y, p in zip(labels_list, decoded_list):
        for gold, st in zip(y, p):
            votes[st][int(gold)] += 1
    return {st: (1 if keep >= drop else 0) for st, (keep, drop) in votes.items()}
