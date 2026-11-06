"""Leave-one-format-out (LOFO): train on 6 formats, test on the held-out 7th.
Compares feature set v1 (15 dims) vs v2 (+7 format-aware dims), HMM and
logreg. Answers: do standardized-format training + format features fix
unseen-format generalization?"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm_filter import GaussianHMM
from hmm_filter.features import featurize
from baselines.filters import LogRegFilter
from data.formats_gen import make
from benchmarks.run_all import prf

def run(feats_version):
    corpus = make()
    D = 15 if feats_version == 1 else 22
    res = {}
    for held in corpus:
        train = [d for k, v in corpus.items() if k != held for d in v]
        test = corpus[held]
        Xtr = [np.array(featurize(d["lines"], version=feats_version)) for d in train]
        ytr = [np.array(d["labels"]) for d in train]
        hmm = GaussianHMM(2, D).fit_supervised(Xtr, ytr)
        lr = LogRegFilter(D).fit(Xtr, ytr)
        for name, pred in [("hmm", None), ("logreg", None)]:
            allp, allg = [], []
            for d in test:
                X = np.array(featurize(d["lines"], version=feats_version))
                p = hmm.viterbi(X)[0] if name == "hmm" else np.array(lr.predict([X])[0])
                allp += [int(x) for x in p]; allg += d["labels"]
            res.setdefault(name, {})[held] = prf(allp, allg)["keep_f1"]
    return res

if __name__ == "__main__":
    out = {}
    for v in (1, 2):
        out[f"v{v}"] = run(v)
        for name in ("hmm", "logreg"):
            f1s = list(out[f"v{v}"][name].values())
            print(f"v{v} {name}: mean_LOFO_F1={np.mean(f1s):.3f} per-format={ {k: round(x,3) for k,x in out[f'v{v}'][name].items()} }", flush=True)
    json.dump(out, open("results/lofo.json", "w"), indent=2)
    print("DONE lofo")
