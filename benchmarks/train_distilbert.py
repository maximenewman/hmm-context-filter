"""Distilled line-classifier baseline: frozen distilbert [CLS] embeddings +
logistic head. RAM-safe (no optimizer states, forward-only embedding passes,
chunked with disk cache). Honest stand-in for the LLMLingua-2 approach at
line level."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModel, AutoTokenizer

HOLDOUT = {"docker", "kubectl", "csv"}
NAME = "distilbert-base-uncased"
EMB = "results/emb_cache.npz"


def load_lines():
    from data.tooltrace_gen import make_corpus
    tt = make_corpus()
    train, test = [], []
    for fam, docs in tt.items():
        for d in docs:
            for ln, y in zip(d["lines"], d["labels"]):
                (test if fam in HOLDOUT else train).append((ln, y))
    return train[:5000], test[:4000]


def embed(start, n):
    train_lines, test_lines = load_lines()
    from data.formats_gen import make as make_formats
    fmt = make_formats()
    fmt_lines = [(l, y, k) for k, docs in fmt.items() for d in docs[:60] for l, y in zip(d["lines"], d["labels"])]
    all_lines = train_lines + test_lines + [(l, y) for l, y, _ in fmt_lines]
    n_tr, n_te = len(train_lines), len(test_lines)
    jobs = list(range(start, min(start + n, len(all_lines))))
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModel.from_pretrained(NAME)
    model.eval()
    out = {}
    if os.path.exists(EMB):
        z = np.load(EMB)
        out = {k: z[k] for k in z.files}
    t0 = time.time()
    for i0 in range(0, len(jobs), 64):
        batch = [all_lines[i][0] for i in jobs[i0:i0 + 64]]
        enc = tok(batch, truncation=True, max_length=48, padding=True, return_tensors="pt")
        with torch.no_grad():
            h = model(**enc).last_hidden_state[:, 0]  # [CLS]
        for j, i in enumerate(jobs[i0:i0 + 64]):
            out[str(i)] = h[j].numpy()
        del enc, h
    np.savez_compressed(EMB, **out)
    print(f"embedded {len(jobs)} lines ({start}..{start+len(jobs)}) in {time.time()-t0:.0f}s, cache={len(out)}", flush=True)


def fit_eval():
    from benchmarks.run_all import prf
    train_lines, test_lines = load_lines()
    from data.formats_gen import make as make_formats
    fmt = make_formats()
    fmt_groups = {k: [(l, y) for d in docs[:60] for l, y in zip(d["lines"], d["labels"])]
                  for k, docs in fmt.items()}
    n_tr = len(train_lines)
    z = np.load(EMB)
    Xtr = np.stack([z[str(i)] for i in range(n_tr)])
    ytr = np.array([y for _, y in train_lines])
    Xte = np.stack([z[str(n_tr + i)] for i in range(len(test_lines))])
    yte = np.array([y for _, y in test_lines])
    # logistic head, numpy
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    w = np.zeros(Xtr.shape[1]); b = 0.0
    Xs = (Xtr - mu) / sd
    pos_w = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    for _ in range(200):
        p = 1 / (1 + np.exp(-(Xs @ w + b)))
        g = (p - ytr) * np.where(ytr == 1, pos_w, 1.0) / len(ytr)
        w -= 0.5 * (Xs.T @ g); b -= 0.5 * g.sum()
    def pred(X):
        return (1 / (1 + np.exp(-(((X - mu) / sd) @ w + b))) > 0.5).astype(int)
    m = prf(pred(Xte).tolist(), yte.tolist())
    print("distilled holdout_families:", m, flush=True)
    off = n_tr + len(test_lines)
    lofo = {}
    i = 0
    for k, lines in fmt_groups.items():
        Xf = np.stack([z[str(off + j)] for j in range(i, i + len(lines))])
        lofo[k] = prf(pred(Xf).tolist(), [y for _, y in lines])["keep_f1"]
        i += len(lines)
    print("distilled lofo_zero_shot:", {k: round(v, 3) for k, v in lofo.items()},
          "mean:", round(float(np.mean(list(lofo.values()))), 3), flush=True)
    json.dump({"holdout": m, "lofo": lofo}, open("results/distilbert.json", "w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        embed(int(sys.argv[2]), int(sys.argv[3]))
    else:
        fit_eval()
