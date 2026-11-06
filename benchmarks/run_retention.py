"""Retention + downstream QA vs compression level.

Part A (fast): evidence-retention curves - at keep-rates {0.9..0.3}, what
fraction of gold answer sentences survives? HMM (keep-posterior threshold)
vs perplexity (drop lowest-loss lines, LLMLingua-style) vs random.

Part B (slow): downstream QA exact-match with Qwen2.5-0.5B-Instruct on
filtered contexts. Chunked: --em-start/--em-n accumulate into results/qa_em.jsonl.
"""
import json, os, sys, re, string
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmm_filter import GaussianHMM, featurize
from data.tooltrace_gen import make_corpus

HOLDOUT = {"docker", "kubectl", "csv"}
LEVELS = [0.9, 0.7, 0.5, 0.3]


def load_hmm():
    tt = make_corpus()
    train = [d for k, v in tt.items() if k not in HOLDOUT for d in v]
    web = json.load(open("data/web_corpus.json"))
    rng = np.random.default_rng(7)
    idxw = rng.permutation(len(web))
    web_train = [web[i] for i in idxw[: len(idxw) // 2]]
    X = [np.array(featurize(d["lines"])) for d in train + web_train]
    y = [np.array(d["labels"]) for d in train + web_train]
    return GaussianHMM(2, 15).fit_supervised(X, y)


def hmm_keep_mask(hmm, lines, rate):
    X = np.array(featurize(lines))
    post = hmm.forward_posteriors(X)[:, 1]
    thr = np.quantile(post, 1 - rate)
    return (post >= thr).astype(int).tolist()


def ppl_keep_mask(pf, lines, rate):
    losses = pf.line_losses(lines)
    order = np.argsort(losses)[::-1]  # highest loss (most surprising) = keep
    keep = [0] * len(lines)
    for i in order[: int(len(lines) * rate)]:
        keep[i] = 1
    return keep


def rand_keep_mask(lines, rate, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(len(lines)) < rate).astype(int).tolist()


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def main():
    squad = json.load(open("data/squad_corpus.json"))
    mode = sys.argv[1] if len(sys.argv) > 1 else "retention"
    hmm = load_hmm()

    if mode == "retention":
        from baselines.filters import PerplexityFilter
        pf = PerplexityFilter(max_len=512)
        docs = squad[:150]
        # score each doc ONCE per method, reuse across levels
        scores = {"hmm": [], "perplexity": [], "random": []}
        for d in docs:
            X = np.array(featurize(d["lines"]))
            scores["hmm"].append(hmm.forward_posteriors(X)[:, 1])
            scores["perplexity"].append(np.array(pf.line_losses(d["lines"])))
            scores["random"].append(np.random.default_rng(0).random(len(d["lines"])))
        print("scores cached", flush=True)
        out = {}
        for name in scores:
            curve = {}
            for rate in LEVELS:
                kept_gold = tot_gold = kept_tok = tot_tok = 0
                for d, sc in zip(docs, scores[name]):
                    if name == "perplexity":
                        thr = np.quantile(sc, 1 - rate); mask = sc >= thr      # high loss = keep
                    else:
                        thr = np.quantile(sc, 1 - rate); mask = sc >= thr      # high score = keep
                    for m, g, ln in zip(mask, d["labels"], d["lines"]):
                        tot_gold += g; kept_gold += g * int(m)
                        t = len(ln.split())
                        tot_tok += t; kept_tok += t * int(m)
                curve[str(rate)] = {"evidence_retention": round(kept_gold / max(tot_gold, 1), 4),
                                    "token_keep": round(kept_tok / max(tot_tok, 1), 4)}
                print(name, rate, curve[str(rate)], flush=True)
            out[name] = curve
        json.dump(out, open("results/retention_curves.json", "w"), indent=2)
        print("DONE retention")

    elif mode == "em":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        start, n = int(sys.argv[2]), int(sys.argv[3])
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float16)
        model.eval()
        data = json.load(open("data/squad_qa.json"))
        jobs = []
        for di, d in enumerate(data):
            for rate in [1.0, 0.7, 0.5, 0.3]:
                jobs.append((di, rate))
        jobs = jobs[start:start + n]
        fo = open("results/qa_em.jsonl", "a")
        for di, rate in jobs:
            d = data[di]
            filt = sys.argv[4] if len(sys.argv) > 4 else "hmm"
            if filt == "random" and rate < 1.0:
                mask = rand_keep_mask(d["lines"], rate)
            elif rate >= 1.0:
                continue
            else:
                mask = hmm_keep_mask(hmm, d["lines"], rate)
            ctx = "\n".join(ln for ln, m in zip(d["lines"], mask) if m)
            msgs = [{"role": "user", "content":
                     f"Context:\n{ctx}\n\nQuestion: {d['question']}\nAnswer with the shortest possible phrase."}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").input_ids
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=24, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            em = max(1.0 if norm(g) in norm(ans) else 0.0 for g in d["answers"])
            fo.write(json.dumps({"doc": di, "rate": rate, "filter": filt, "em": em,
                                 "ans": ans[:80], "gold": d["answers"][0][:80]}) + "\n")
            fo.flush()
            print(di, rate, em, ans[:40].replace("\n", " "), flush=True)
        print("DONE chunk", start, n)


if __name__ == "__main__":
    main()
