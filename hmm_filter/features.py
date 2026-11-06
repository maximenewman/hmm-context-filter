"""Cheap per-line features for the HMM stream filter.

Every feature is O(len(line)) with a tiny constant. No tokenization, no
model calls, no I/O. This is the whole point of the project: the filter
must cost ~0 tokens and ~0 latency relative to the LLM it protects.
"""
import math
import string

_PRINTABLE = set(string.printable)
_NOISE_WORDS = (
    "error", "warn", "debug", "traceback", "exception", "failed",
    "retry", "deprecated", "notice:", "verbose", "stacktrace",
)
_BOILER_HINTS = (
    "copyright", "license", "all rights reserved", "terms of service",
    "privacy policy", "cookie", "subscribe", "sign up", "log in",
    "accept", "navigation", "menu", "footer", "powered by",
)


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def line_features(line: str, prev_line: str = "", version: int = 2) -> list[float]:
    """Feature vector for one line. 12 dims, all float, roughly scaled."""
    n = len(line)
    if n == 0:
        return [0.0] * (15 if version == 1 else 22)
    letters = sum(c.isalpha() for c in line)
    digits = sum(c.isdigit() for c in line)
    spaces = sum(c.isspace() for c in line)
    uppers = sum(c.isupper() for c in line)
    puncts = sum(c in string.punctuation for c in line)
    low = line.lower()
    stripped = line.strip()
    indent = n - len(line.lstrip())
    words = [w for w in stripped.split() if any(c.isalpha() for c in w)]
    word_lens = [len(w.strip(string.punctuation)) for w in words]
    return [
        math.log1p(n),                    # 0 log length
        letters / n,                      # 1 alpha ratio
        digits / n,                       # 2 digit ratio
        spaces / n,                       # 3 whitespace ratio
        puncts / n,                       # 4 punctuation ratio
        uppers / n,                       # 5 uppercase ratio
        shannon_entropy(line),            # 6 char entropy
        sum(w in low for w in _NOISE_WORDS),      # 7 noise-word hits
        sum(w in low for w in _BOILER_HINTS),     # 8 boilerplate hits
        1.0 if stripped and stripped == prev_line.strip() else 0.0,  # 9 repeat
        min(indent, 32) / 32.0,           # 10 indentation
        1.0 if stripped.endswith(("{", "[", ":", ",", "}")) else 0.0,  # 11 structural
        len(words) / max(n / 6.0, 1.0),   # 12 word density (prose ~1)
        (sum(word_lens) / max(len(word_lens), 1)) / 10.0,  # 13 avg word len
        1.0 if stripped.endswith((".", "?", "!", '"')) else 0.0,  # 14 sentence-final
    ] + _format_features(line, stripped, n)


def _format_features(line: str, stripped: str, n: int) -> list[float]:
    """v2: standardized-format structure. Tool-agnostic by design."""
    import re as _re
    json_struct = 1.0 if _re.match(r'^[\{\}\[\]],?$|^"[\w.-]+"\s*:', stripped) else 0.0
    kv = 1.0 if _re.match(r'^[\w."-]+\s*[:=]\s*\S+', stripped) else 0.0
    md = 1.0 if _re.match(r'^(#{1,6}\s|[-*>]\s|\||```|\[)!', stripped) else 0.0
    tags = len(_re.findall(r"<[^>]+>", line))
    quotes = line.count('"') / max(n, 1)
    long_tok = 1.0 if any(len(t) >= 16 and t.isalnum() for t in stripped.split()) else 0.0
    comment = 1.0 if stripped.startswith(("#", "//", "<!--", ";")) and not stripped.startswith("# ") else 0.0
    return [json_struct, kv, md, min(tags, 4) / 4.0, quotes, long_tok, comment]


FEATURE_NAMES = [
    "log_len", "alpha", "digit", "space", "punct", "upper",
    "entropy", "noise_words", "boiler_words", "repeat", "indent", "struct",
    "word_density", "avg_word_len", "sent_final",
]


def featurize(lines: list[str], version: int = 2) -> "list[list[float]]":
    out = []
    prev = ""
    for ln in lines:
        f = line_features(ln, prev, version=version)
        out.append(f if version == 2 else f[:15])
        prev = ln
    return out
