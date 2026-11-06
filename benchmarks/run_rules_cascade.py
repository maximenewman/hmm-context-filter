"""Rule-based hand filter (Isthmos-style: what an engineer writes today)
and the cascade test: HMM stage-0 -> perplexity stage-1, retention vs cost."""
import json, os, sys, re, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm_filter import GaussianHMM, featurize
from data.tooltrace_gen import make_corpus
from data.formats_gen import make as make_formats
from benchmarks.run_all import prf

DROP_PATTERNS = [
    r"\[[=\s]{5,}\]\s*\d+%",          # progress bars
    r"Downloading|Collecting|eta \d",  # pip spam
    r"--->\s",                         # docker cache lines
    r"^[0-9a-f]{12,}$",                # bare hashes
    r"^(#|//|;|<!--)",                 # comments
    r"badge|shields\.io|Table of contents",
    r"heartbeat",
    r"^[\s{}\[\],]*$",                 # pure structural chars
    r"^(Date:|Content-Type:|X-[A-Z]|Cache-Control:)",  # HTTP headers
    r"Copyright|All rights reserved|Privacy Policy|Terms of Service",
]
RULES = [re.compile(p) for p in DROP_PATTERNS]

def rule_filter(lines):
    return [0 if any(r.search(l) for r in RULES) else 1 for l in lines]

def detect_format(lines):
    text = "\n".join(lines[:50])
    if re.search(r"^\s*[{\[]", text, re.M) and '":' in text: return "json"
    if re.search(r"<[a-zA-Z/][^>]*>", text): return "xml"
    if re.search(r"^[\w.-]+=", text, re.M): return "logfmt"
    if re.search(r"^[^,]+,[^,]+,[^,]+$", text, re.M): return "csv"
    if re.search(r"^#{1,6}\s|\]\(", text, re.M): return "markdown"
    if re.search(r"^\s*[\w.-]+:\s", text, re.M): return "yaml"
    return "prose"

HOLDOUT = {"docker", "kubectl", "csv"}

def main():
    tt = make_corpus()
    holdout_docs = [d for k, v in tt.items() if k in HOLDOUT for d in v]
    # rules on tooltrace holdout
    allp = [x for d in holdout_docs for x in rule_filter(d["lines"])]
    allg = [x for d in holdout_docs for x in d["labels"]]
    print("rules_on_tooltrace_holdout:", prf(allp, allg), flush=True)

    # parse-then-filter on LOFO formats: unseen format -> keep all (safe)
    corpus = make_formats()
    safe = {}
    for held in corpus:
        train = [d for k, v in corpus.items() if k != held for d in v]
        test = corpus[held]
        Xtr = [np.array(featurize(d["lines"])) for d in train]
        ytr = [np.array(d["labels"]) for d in train]
        hmm = GaussianHMM(2, 22).fit_supervised(Xtr, ytr)
        trained_fmts = set(corpus) - {held}
        allp, allg = [], []
        for d in test:
            fmt = detect_format(d["lines"])
            if fmt not in trained_fmts:
                p = [1] * len(d["lines"])  # unknown format: keep everything
            else:
                p = [int(x) for x in hmm.viterbi(np.array(featurize(d["lines"])))[0]]
            allp += p; allg += d["labels"]
        safe[held] = prf(allp, allg)["keep_f1"]
    print("parse_then_filter_safe:", {k: round(v,3) for k,v in safe.items()},
          "mean:", round(float(np.mean(list(safe.values()))), 3), flush=True)

    # cascade on squad: hmm keeps 70% -> ppl trims to 50% of original
    from baselines.filters import PerplexityFilter
    squad = json.load(open("data/squad_corpus.json"))[:60]
    web = json.load(open("data/web_corpus.json"))
    rng = np.random.default_rng(7); idxw = rng.permutation(len(web))
    train_docs = [d for k, v in tt.items() if k not in HOLDOUT for d in v] + [web[i] for i in idxw[:len(idxw)//2]]
    hmm = GaussianHMM(2, 22).fit_supervised(
        [np.array(featurize(d["lines"])) for d in train_docs],
        [np.array(d["labels"]) for d in train_docs])
    pf = PerplexityFilter(max_len=512)

    def retention(masks):
        kg = tg = 0
        for d, m in zip(squad, masks):
            for mi, g in zip(m, d["labels"]):
                tg += g; kg += g * mi
        return kg / max(tg, 1)

    t0 = time.time(); masks_hmm = []
    for d in squad:
        post = hmm.forward_posteriors(np.array(featurize(d["lines"])))[:, 1]
        thr = np.quantile(post, 0.3)
        masks_hmm.append((post >= thr).astype(int))
    t_hmm = time.time() - t0

    t0 = time.time(); masks_ppl = []
    for d in squad:
        losses = np.array(pf.line_losses(d["lines"]))
        thr = np.quantile(losses, 0.5)
        masks_ppl.append((losses >= thr).astype(int))
    t_ppl = time.time() - t0

    t0 = time.time(); masks_casc = []
    for d, m1 in zip(squad, masks_hmm):
        surv = [i for i, x in enumerate(m1) if x]
        sub = [d["lines"][i] for i in surv]
        if not sub:
            masks_casc.append(m1); continue
        losses = np.array(pf.line_losses(sub))
        keep_n = int(len(d["lines"]) * 0.5)
        order = np.argsort(losses)[::-1][:keep_n]
        m = np.zeros(len(d["lines"]), dtype=int)
        for oi in order:
            m[surv[oi]] = 1
        masks_casc.append(m)
    t_casc = time.time() - t0 + t_hmm

    print(f"retention@50% hmm-only={retention(masks_hmm):.3f} ppl-only={retention(masks_ppl):.3f} "
          f"cascade={retention(masks_casc):.3f}", flush=True)
    print(f"wall_time hmm={t_hmm:.2f}s ppl={t_ppl:.2f}s cascade={t_casc:.2f}s", flush=True)
    json.dump({"rules_tooltrace_holdout": prf(allp, allg),
               "parse_then_filter_safe": safe,
               "cascade": {"hmm_only": retention(masks_hmm), "ppl_only": retention(masks_ppl),
                            "cascade": retention(masks_casc),
                            "t_hmm": t_hmm, "t_ppl": t_ppl, "t_cascade": t_casc}},
              open("results/rules_cascade.json", "w"), indent=2)
    print("DONE rules+cascade")

main()
