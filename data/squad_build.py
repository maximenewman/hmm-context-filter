"""SQuAD-based answer-retention corpus.

For each paragraph, sentences containing a gold answer span are keep=1,
everything else drop=0. A filter is good if it compresses a lot while
retaining the answer sentence. This mirrors the LLMLingua/EffComp eval
style (does the evidence survive compression?) without needing an LLM judge.
"""
import json, re, random

def sentence_spans(text):
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text):
        spans.append((start, m.start())); start = m.end()
    spans.append((start, len(text)))
    return [(s, e, text[s:e].strip()) for s, e in spans if text[s:e].strip()]

def build(path="data/dev-v1.1.json", sample=600, seed=0):
    data = json.load(open(path))
    rng = random.Random(seed)
    paras = [p for a in data["data"] for p in a["paragraphs"]]
    rng.shuffle(paras)
    corpus = []
    for p in paras:
        ctx = p["context"]
        answers = [qa["answers"] for qa in p["qas"] if qa.get("answers")]
        if not answers:
            continue
        spans = sentence_spans(ctx)
        if len(spans) < 4:
            continue
        labels = [0] * len(spans)
        for ans_set in answers:
            for ans in ans_set:
                a0 = ans["answer_start"]
                for i, (s, e, _) in enumerate(spans):
                    if s <= a0 < e:
                        labels[i] = 1
        if sum(labels) == 0:
            continue
        corpus.append({"lines": [t for _, _, t in spans], "labels": labels})
        if len(corpus) >= sample:
            break
    return corpus

if __name__ == "__main__":
    c = build()
    json.dump(c, open("data/squad_corpus.json", "w"))
    tot = sum(len(x["lines"]) for x in c)
    keep = sum(sum(x["labels"]) for x in c)
    print(f"paras={len(c)} sentences={tot} keep_rate={keep/tot:.2f}")
