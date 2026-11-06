# hmm-context-filter

A hidden Markov model that filters agent tool outputs and long context
**before** they reach the LLM. It reads text line by line and tags each line
as keep or drop, costing effectively zero tokens and microseconds per line.
The pitch: everything else that filters context (LLMLingua, EffComp, an LLM
judge) pays model compute to do the filtering. This pays almost nothing.

## What this is

- `hmm_filter/` - the filter. 15 cheap features per line (length, entropy,
  digit/punctuation ratios, keyword hits, repetition, indentation, prose
  shape), a diagonal-Gaussian HMM over hidden states (useful / boilerplate /
  noise), Viterbi decoding offline, forward-filtering for true streaming.
  numpy only. The trained model is 1.5 KB.
- `data/` - three corpora with ground-truth keep/drop labels:
  1. **tooltrace** - 1,200 synthetic agent tool outputs (pip, HTTP, git,
     search, pytest, docker, stack traces, HTML, kubectl, CSV), line labels
     known by construction.
  2. **web** - 29 real web pages (Wikipedia, docs, news), labels derived
     from the HTML structure (nav/footer/aside = drop, article text = keep).
  3. **squad** - 600 SQuAD dev paragraphs; sentences containing a gold
     answer are keep, the rest drop. Measures evidence retention.
- `baselines/` - keep-all, random, length heuristic, logistic regression on
  the same features (ablation), and a document-conditioned perplexity
  filter using distilgpt2 (the LLMLingua idea, small).
- `benchmarks/` - the runners. `run_all.py` reproduces every number in
  `results/classification.json`. `provider_harness.py` estimates (and, with
  keys and approval, runs) the Claude/GPT/Gemini comparison.

## Headline numbers (this repo, reproducible)

See `results/classification.json` for the full table. Summary:

- On tool traces (random split): the HMM removes ~65% of tokens while
  keeping ~2/3 of useful lines (keep-F1 0.63). A logistic regression on the
  same features gets 0.70 - so per-line structure matters more than sequence
  here. Both are honest, small numbers; this is a pre-filter, not magic.
- On real web pages, an HMM trained on tool traces + half the web corpus
  keeps 93% of content lines while dropping 76% of tokens (keep-F1 0.82).
- Speed: ~55,000 lines/sec on one CPU core vs ~70 lines/sec for the
  distilgpt2 perplexity filter - about 800x cheaper per line, before you
  count that distilgpt2 itself is 1,000x cheaper than a frontier LLM call.
- Honest failure: trained on 7 tool families and tested on 3 UNSEEN
  families (docker, kubectl, CSV), every stylistic filter collapses
  (HMM keep-F1 0.10). Content in an unfamiliar format looks like noise.
  Lesson: train the filter on the tool formats you actually run.
- Honest failure 2: on SQuAD evidence sentences the filter (correctly)
  keeps ~99% - prose evidence has no stylistic signature. Query-aware
  neural compressors (LLMLingua-2, EffComp) exist for a reason.

## Prior work and how this differs

- LLMLingua (EMNLP 2023) / LLMLingua-2 (ACL 2024): LLM/token-classifier
  compression. Strong, task-agnostic, but pays a model per compression.
- EffComp (arXiv 2611.02766, 2026): RL+SFT extractive compression for RAG.
- TACO-RL (2024): RL for prompt compression.
- Conroy & O'Leary (SIGIR 2001): HMM sentence selection for summarization.
  The ancestor. We differ by targeting streaming tool traces, not document
  summaries, and by being an always-on pre-filter, not a one-shot summary.
- Nobody we found ships an HMM pre-filter for agent tool traces. "No
  published work found" is not proof of first.

## Reproduce

```
pip install -r requirements.txt
python3 data/tooltrace_gen.py
python3 data/web_build.py        # refetches 29 pages
python3 data/squad_build.py      # needs data/dev-v1.1.json (SQuAD dev)
python3 benchmarks/run_all.py fast
python3 benchmarks/run_all.py ppl-squad   # etc, per corpus
```

## Status

Research prototype, built Nov 2026. Numbers are real and reproducible;
claims are narrow on purpose.


## Results (all reproducible from this repo)

### Keep/drop classification (results/classification.json)

| corpus | filter | keep-F1 | tokens kept | lines/sec |
|---|---|---|---|---|
| tool traces (random split) | HMM | 0.63 | 35% | 48,853 |
| tool traces (random split) | logreg ablation | 0.70 | 22% | 58,526 |
| tool traces (random split) | length heuristic | 0.50 | 73% | - |
| tool traces (unseen families) | HMM | 0.10 | 18% | - |
| tool traces (unseen families) | perplexity (distilgpt2) | 0.36 | 26% | 69 |
| web pages (zero-shot) | HMM (tool-trained) | 0.34 | 20% | - |
| web pages | HMM (multi-domain) | 0.82 | 24% | 37,398 |
| web pages | perplexity (distilgpt2) | 0.61 | 61% | 1,862 |
| SQuAD prose | any stylistic filter | ~random | - | - |

### The generalization question, answered (results/lofo.json)

User hypothesis: train on standardized formats (JSON, XML, markdown,
logfmt, CSV, YAML, prose) instead of per-tool. Leave-one-format-out:

| model | mean LOFO keep-F1 | note |
|---|---|---|
| HMM, base features | 0.487 | JSON 0.19, CSV 0.00 |
| HMM, +7 format features | 0.504 | JSON 0.61, CSV still 0.00 |
| logreg, +7 format features | 0.569 | JSON 0.75, CSV still 0.00 |
| parse-then-filter (safe fallback) | 0.559 | CSV 0.86, nothing ever lost |
| distilled distilbert + logreg head (zero-shot here) | 0.749 | CSV 0.81 - semantics generalize |


Verdict: format-aware training helps when formats share structure (JSON
jumped 0.19 -> 0.61), but a held-out format with no structural analogue
(CSV data rows) still collapses for STYLISTIC filters. The frozen-distilbert
line classifier solves it (0.75 mean, CSV 0.81) - semantic features
generalize - at ~1000x the per-line cost (51 vs 55,000 lines/sec) and it
still loses to hand rules on unseen tool families (0.42 vs 0.75 keep-F1,
results/distilbert.json). The robust answer is the user's second
idea: parse first, then filter - detect the format, apply the per-format
policy, and keep everything when the format is unknown. That trades
compression for zero data loss on surprises, which is the right default.

### Cascade and downstream (results/rules_cascade.json, results/qa_em.jsonl)

- Rule-based hand filter (13 regexes, the thing an engineer writes today):
  keep-F1 0.75 on UNSEEN tool families vs the HMM's 0.10. Hand rules encode
  structure that transfers (progress bars look the same everywhere); the
  learned filter's features don't. On SEEN formats the learned filter wins
  and compresses 3x harder. Honest synthesis: ship rules + HMM together.
- Cascade HMM -> perplexity on SQuAD: retention@50% 0.457 vs ppl-only
  0.560 vs hmm-only@70% 0.659. On clean prose the stage-0 errors dominate;
  a cascade only pays on noisy streams where stage-0 recall is ~1.
- Downstream QA (Qwen2.5-0.5B, 60 SQuAD paras): EM 0.55 full context,
  0.25 at 70% keep, 0.18 at 30% keep. Filtering prose evidence destroys
  reader accuracy faster than it saves tokens. Do not filter QA evidence
  with a stylistic filter.

### Speed

~55k lines/sec/core (HMM) vs ~70 lines/sec (distilgpt2 perplexity) vs
~$2-4 per 1M tokens (frontier LLM as filter). The HMM model file is 1.5 KB.

## Honest positioning

This is a learned, zero-cost **stage 0**: a bouncer that strips obvious
noise from noisy tool streams before anything expensive runs. It is NOT an
LLMLingua replacement - LLMLingua-2/EffComp do query-aware semantic
compression of prose, which this structurally cannot do. The two compose:
stage 0 (this) removes garbage at ~0 cost, stage 1 (neural) does semantic
compression on what survives, on corpora where that pays.

## Prior work

- Conroy & O'Leary 2001 (HMM sentence selection for summarization) - the ancestor.
- LLMLingua (arXiv 2310.05736): up to 20x compression, GSM8K/BBH/ShareGPT.
- LLMLingua-2 (arXiv 2403.12968): 2x-5x compression, 3-6x faster than LLMLingua.
- EffComp (arXiv 2611.02766): BERT reranker + RL/SFT selector for RAG QA.
- TACO-RL 2024, Selective-Context 2023: RL / self-information compression.
- Isthmos (hand rules), MCP+ (cheap-LLM extraction): practitioner points
  in the cost/quality space; hand rules are baseline above, cheap-LLM
  extraction is what provider_harness.py measures.

## Not done yet / future work

- CRF baseline (between HMM and distilbert in the cost/quality space).
- Fine-tuned (not frozen) distilbert; distilled model trained on the
  format corpus too (here it was zero-shot on formats and still won).
- Loghub/Drain real-log corpus; MCP-Universe end-to-end replay
  (task success, tokens saved p50/p90, expand rate) - needs real agent
  sessions, which this sandbox does not have.
- Provider comparison (Claude Opus 5.5 / GPT-5.1 / Gemini 3.1 Pro) is
  wired in benchmarks/provider_harness.py, ~$15 for a full pass, waiting
  on API keys and approval.
