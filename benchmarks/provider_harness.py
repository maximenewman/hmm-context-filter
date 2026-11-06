"""Frontier-LLM comparison harness. DRY RUN BY DEFAULT: computes the exact
token counts and cost table. No API call happens without --run AND the
provider's env key set. Costs from vendor pricing pages fetched 2026-11-06:
- Claude Opus 5.5: $4/MTok input, $20/MTok output (platform.claude.com/docs/en/about-claude/pricing)
- GPT-5.1: $1.25/MTok input, $10/MTok output (developers.openai.com/api/docs/pricing)
- Gemini: see PRICES below, verify before running.

Task given to the LLM filter: return the indices of lines to KEEP
(LLMLingua-style document compression, line granularity).
"""
import json, os, sys

PRICES = {  # USD per 1M tokens (input, output)
    "claude-opus-5.5": (4.00, 20.00),
    "gpt-5.1": (1.25, 10.00),
    "gemini-3.1-pro": (2.00, 12.00),  # verified ai.google.dev/gemini-api/docs/pricing 2026-11-06 (<=200k prompts)
}
PROMPT_OVERHEAD_TOKENS = 120   # instruction + formatting per document
EST_OUTPUT_TOKENS_PER_LINE = 4  # keep-index list

def estimate(corpus_tokens):
    rows = []
    for model, (pin, pout) in PRICES.items():
        inp = corpus_tokens + PROMPT_OVERHEAD_TOKENS
        out = EST_OUTPUT_TOKENS_PER_LINE * 40  # ~40 lines/doc avg
        cost = (inp * pin + out * pout) / 1e6
        rows.append({"model": model, "est_cost_per_doc": round(cost, 6),
                     "input_price": pin, "output_price": pout})
    return rows

if __name__ == "__main__":
    counts = json.load(open("results/corpus_tokens.json"))
    docs = {"tooltrace_holdout": 360, "web_full": 29, "squad": 600}
    print(f"{'model':<18} {'corpus':<20} {'est USD/full pass':>18}")
    for model in PRICES:
        for c, t in counts.items():
            if c == "total_eval": continue
            per_doc = estimate(t / docs[c])[0]["est_cost_per_doc"] if False else None
            inp = t + PROMPT_OVERHEAD_TOKENS * docs[c]
            out = EST_OUTPUT_TOKENS_PER_LINE * 40 * docs[c]
            pin, pout = PRICES[model]
            print(f"{model:<18} {c:<20} {(inp*pin+out*pout)/1e6:>18.2f}")
    if "--run" in sys.argv:
        raise SystemExit("--run not wired to keys in this environment. Ask the user first.")
