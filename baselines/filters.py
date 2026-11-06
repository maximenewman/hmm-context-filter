"""Baselines: keep-all, random, length heuristic, perceptron-style logreg,
and a document-conditioned perplexity filter (LLMLingua-style, distilgpt2).
All expose: predict(lines) -> list[int] (1=keep, 0=drop), or scores."""
import numpy as np


def keep_all(lines):
    return [1] * len(lines)


def random_drop(lines, rate, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(len(lines)) > rate).astype(int).tolist()


def length_heuristic(lines, min_len=40):
    return [1 if len(l) >= min_len else 0 for l in lines]


class LogRegFilter:
    """Same features as the HMM, no sequential structure. Ablation."""
    def __init__(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0.0

    def fit(self, X_list, y_list, epochs=60, lr=0.05):
        X = np.vstack(X_list); y = np.concatenate(y_list).astype(float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        self.mu, self.sd = mu, sd
        X = (X - mu) / sd
        for _ in range(epochs):
            z = X @ self.w + self.b
            p = 1 / (1 + np.exp(-z))
            g = (p - y) / len(y)
            self.w -= lr * (X.T @ g)
            self.b -= lr * g.sum()
        return self

    def predict(self, X_list):
        X = np.vstack(X_list)
        z = ((X - self.mu) / self.sd) @ self.w + self.b
        out = (1 / (1 + np.exp(-z)) > 0.5).astype(int)
        res, i = [], 0
        for x in X_list:
            res.append(out[i:i + len(x)].tolist()); i += len(x)
        return res


class PerplexityFilter:
    """LLMLingua-style: token-level causal-LM loss over the whole document,
    averaged per line; lines with the LOWEST mean loss (most predictable)
    are dropped first. Uses distilgpt2 on CPU."""
    def __init__(self, model_name="distilgpt2", max_len=1000):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.torch = torch
        self.max_len = max_len

    def line_losses(self, lines):
        """Mean per-token loss attributed to each line (document context)."""
        torch = self.torch
        offsets, enc_lines = [], []
        pos = 0
        full = "\n".join(lines)
        enc = self.tok(full, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"][: self.max_len]
        offs = enc["offset_mapping"][: self.max_len]
        # line start offsets in the joined text
        starts, acc = [], 0
        for ln in lines:
            starts.append(acc); acc += len(ln) + 1
        import bisect
        line_of_tok = [bisect.bisect_right(starts, s) - 1 for s, _ in offs]
        with torch.no_grad():
            out = self.model(torch.tensor([ids]), labels=torch.tensor([ids]))
        logits = out.logits[0]
        logp = torch.log_softmax(logits, dim=-1)
        losses = []
        for t in range(1, len(ids)):
            losses.append(-logp[t - 1, ids[t]].item())
        tok_line = line_of_tok[1:]
        per_line = {}
        for lo, li in zip(losses, tok_line):
            per_line.setdefault(li, []).append(lo)
        return [float(np.mean(per_line.get(i, [3.0]))) for i in range(len(lines))]

    def predict(self, lines, drop_frac=0.5):
        losses = self.line_losses(lines)
        k = int(len(lines) * drop_frac)
        order = np.argsort(losses)  # lowest loss = most predictable = drop first
        keep = [1] * len(lines)
        for i in order[:k]:
            keep[i] = 0
        return keep
