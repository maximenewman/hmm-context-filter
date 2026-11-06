"""Main benchmark: HMM stream filter vs baselines on 3 corpora.

Splits:
- tooltrace: train on 7 tool families, test on 3 HELD-OUT families
  (generalization to unseen tools) + a random-split control.
- web: page-level random split (silver DOM labels; HMM trained on
  tooltrace is also tested zero-shot here - the cross-domain claim).
- squad: paragraph split; retention of answer-bearing sentences vs
  compression.

Outputs results/*.json and results/tables.md.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm_filter import GaussianHMM, featurize
from hmm_filter.hmm import state_keep_map
from data.tooltrace_gen import make_corpus
from baselines.filters import (keep_all, random_drop, length_heuristic,
                               LogRegFilter, PerplexityFilter)

HOLDOUT = {"docker", "kubectl", "csv"}


def prf(pred, gold):
    pred, gold = np.asarray(pred), np.asarray(gold)
    tp = int(((pred == 1) & (gold == 1)).sum()); fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum()); tn = int(((pred == 0) & (gold == 0)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {"keep_p": round(p, 4), "keep_r": round(r, 4), "keep_f1": round(f1, 4), "acc": round(acc, 4)}


def compress_ratio(docs, preds, tok):
    kept = tot = 0
    for d, p in zip(docs, preds):
        for ln, k in zip(d["lines"], p):
            n = len(tok(ln)["input_ids"])
            tot += n; kept += n * k
    return round(kept / max(tot, 1), 4)


def eval_filter(name, predict_fn, test_docs, tok):
    t0 = time.time()
    preds = [predict_fn(d["lines"]) for d in test_docs]
    dt = time.time() - t0
    allp = [x for p in preds for x in p]; allg = [x for d in test_docs for x in d["labels"]]
    m = prf(allp, allg)
    m["token_keep_ratio"] = compress_ratio(test_docs, preds, tok)
    n_lines = len(allg)
    m["lines_per_s"] = round(n_lines / max(dt, 1e-6))
    return name, m, preds


def three_way_labels(labels, lines):
    """Pseudo 3-class: 2=keep, 1=boilerplate-drop, 0=noise-drop (heuristic:
    drops containing error/debug vocabulary are 'noise')."""
    noise_words = ("error", "warn", "debug", "traceback", "exception", "failed", "%]", "eta")
    out = []
    for labs, lns in zip(labels, lines):
        y = []
        for g, ln in zip(labs, lns):
            if g == 1:
                y.append(2)
            else:
                y.append(0 if any(w in ln.lower() for w in noise_words) else 1)
        out.append(np.array(y))
    return out


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "fast"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    results = {}

    tt = make_corpus()
    train_fams = {k: v for k, v in tt.items() if k not in HOLDOUT}
    test_fams = {k: v for k, v in tt.items() if k in HOLDOUT}
    train_docs = [d for v in train_fams.values() for d in v]
    holdout_docs = [d for v in test_fams.values() for d in v]
    # random-split control within all families
    rng = np.random.default_rng(7)
    all_docs = [d for v in tt.values() for d in v]
    idx = rng.permutation(len(all_docs))
    cut = int(0.7 * len(all_docs))
    rs_train = [all_docs[i] for i in idx[:cut]]; rs_test = [all_docs[i] for i in idx[cut:]]

    web = json.load(open("data/web_corpus.json"))
    idxw = rng.permutation(len(web))
    web_test = [web[i] for i in idxw[len(idxw) // 2:]]
    squad = json.load(open("data/squad_corpus.json"))

    # ---- train HMM 2-state on tooltrace train
    Xtr = [np.array(featurize(d["lines"])) for d in train_docs]
    ytr = [np.array(d["labels"]) for d in train_docs]
    hmm2 = GaussianHMM(2, 15).fit_supervised(Xtr, ytr)
    hmm2_keep = {0: 0, 1: 1}  # supervised: state index IS the label

    # ---- train HMM 3-state with pseudo-labels
    y3 = three_way_labels(ytr, [d["lines"] for d in train_docs])
    hmm3 = GaussianHMM(3, 15).fit_supervised(Xtr, y3)

    # ---- logreg ablation
    lr = LogRegFilter(15).fit(Xtr, ytr)

    def hmm2_vit(lines):
        p, _ = hmm2.viterbi(np.array(featurize(lines)))
        return [hmm2_keep.get(int(s), 1) for s in p]

    def hmm2_online(lines):
        p = hmm2.predict_online(np.array(featurize(lines)))
        return [hmm2_keep.get(int(s), 1) for s in p]

    def hmm3_vit(lines):
        p, _ = hmm3.viterbi(np.array(featurize(lines)))
        return [1 if s == 2 else 0 for s in p]

    def lr_pred(lines):
        return lr.predict([np.array(featurize(lines))])[0]

    # ---- hmm_multi: trained on tooltrace + the train half of web pages
    web_train = [web[i] for i in idxw[: len(idxw) // 2]]
    Xmulti = Xtr + [np.array(featurize(d["lines"])) for d in web_train]
    ymulti = ytr + [np.array(d["labels"]) for d in web_train]
    hmmM = GaussianHMM(2, 15).fit_supervised(Xmulti, ymulti)

    def hmmM_vit(lines):
        p, _ = hmmM.viterbi(np.array(featurize(lines)))
        return [int(s) for s in p]

    def hmmM_online(lines):
        return [int(s) for s in hmmM.predict_online(np.array(featurize(lines)))]

    corpora = {"tooltrace_holdout": holdout_docs, "tooltrace_random": rs_test,
               "web_zero_shot": web_test, "squad": squad}
    filters = {"hmm_viterbi": hmm2_vit, "hmm_online": hmm2_online, "hmm3_viterbi": hmm3_vit,
               "hmm_multi_viterbi": hmmM_vit, "hmm_multi_online": hmmM_online,
               "logreg": lr_pred, "keep_all": keep_all, "length_heuristic": length_heuristic}

    out = "results/classification.json"
    results = json.load(open(out)) if os.path.exists(out) else {}
    if stage == "fast":
        for cname, docs in corpora.items():
            results.setdefault(cname, {})
            for fname, fn in filters.items():
                name, m, _ = eval_filter(fname, fn, docs, tok)
                results[cname][name] = m
                print(cname, name, m, flush=True)
            drop_rate = 1 - np.mean([np.mean(d["labels"]) for d in docs])
            name, m, _ = eval_filter("random", lambda l: random_drop(l, drop_rate), docs, tok)
            results[cname][name] = m
            print(cname, "random", m, flush=True)
        json.dump(results, open(out, "w"), indent=2)
        print("DONE fast")
    else:
        # ppl-<corpus>: perplexity baseline on up to 150 docs, 512-token docs
        cname = stage.split("-", 1)[1]
        pf = PerplexityFilter(max_len=512)
        docs = corpora[cname][:150]
        target = np.mean([np.mean(d["labels"]) for d in docs])
        fn = lambda l: pf.predict(l, drop_frac=1 - target)
        name, m, _ = eval_filter("perplexity_distilgpt2", fn, docs, tok)
        results.setdefault(cname, {})[name] = m
        results[cname]["_ppl_note"] = f"n={len(docs)} docs, 512-token cap, keep-rate matched {target:.3f}"
        print(cname, name, m, flush=True)
        json.dump(results, open(out, "w"), indent=2)
        print("DONE", stage)


if __name__ == "__main__":
    main()
