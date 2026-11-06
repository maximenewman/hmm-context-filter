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
