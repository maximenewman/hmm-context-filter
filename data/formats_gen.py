"""Format-pure synthetic corpus for leave-one-format-out (LOFO) evaluation.

Hypothesis under test (user's idea): if the filter learns STANDARD FORMATS
(JSON, XML, markdown, logfmt, CSV, YAML, prose) instead of specific tools,
it should generalize to unseen tools that emit those formats.
Each generator produces (lines, labels); label 1 = keep.
"""
import random, json

CONTENT = [
    "The quarterly revenue grew 14% year over year to 2.3 million dollars.",
    "Migration 0042_alter_users_table applied successfully in 0.83 seconds.",
    "Endpoint /v2/orders accepts POST with fields item_id, quantity and note.",
    "Average response latency dropped from 480ms to 210ms after the cache fix.",
    "Set MAX_CONNECTIONS=64 in production to avoid pool exhaustion.",
    "The committee approved the budget after a two hour debate on Tuesday.",
    "Battery capacity is 4800 mAh with 65W fast charging support.",
    "Our analysis indicates the outage was caused by a misconfigured router.",
]
W = "alpha beta gamma delta cache node proxy shard queue user order item".split()

def json_api(rng):
    lines, labs = ['{'], [0]
    for _ in range(rng.randint(2, 5)):
        lines += ['  {', f'    "id": {rng.randint(1000,9999)},',
                  f'    "type": "{rng.choice(W)}",',
                  f'    "fact": "{rng.choice(CONTENT)}",',
                  f'    "ts": {rng.randint(1600000000,1799999999)}', '  },']
        labs += [0, 0, 0, 1, 0, 0]
    lines += [f'  "page": {rng.randint(1,9)},', '}']; labs += [0, 0]
    return lines, labs

def xml_feed(rng):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<rss version="2.0">', '<channel>']; labs = [0,0,0]
    for _ in range(rng.randint(2, 4)):
        lines += ['  <item>', f'    <title>{rng.choice(CONTENT)}</title>',
                  f'    <link>https://example.com/{rng.choice(W)}/{rng.randint(1,99)}</link>',
                  f'    <guid isPermaLink="false">{rng.randrange(16**8):08x}</guid>', '  </item>']
        labs += [0, 1, 0, 0, 0]
    lines += ['</channel>', '</rss>']; labs += [0, 0]
    return lines, labs

def markdown_doc(rng):
    lines = [f'# {rng.choice(W).title()} Documentation', '', '[![badge](https://img.shields.io/badge.svg)]()',
             '## Table of contents', '- [Install](#install) - [Usage](#usage)']
    labs = [0, 0, 0, 0, 0]
    for _ in range(rng.randint(3, 6)):
        lines += [rng.choice(CONTENT)]; labs += [1]
        if rng.random() < 0.4:
            lines += [f'    {rng.choice(W)} --{rng.choice(W)}=value']; labs += [0]
    lines += ['## License', 'MIT. Copyright 2026.']; labs += [0, 0]
    return lines, labs

def logfmt_events(rng):
    lines, labs = [], []
    for _ in range(rng.randint(5, 10)):
        lvl = rng.choice(['debug', 'debug', 'info', 'error'])
        msg = rng.choice(CONTENT) if lvl == 'error' else f'{rng.choice(W)} heartbeat ok'
        lines += [f'ts=2026-11-06T1{rng.randint(0,9)}:0{rng.randint(0,9)}:00Z level={lvl} svc={rng.choice(W)} msg="{msg}"']
        labs += [1 if lvl == 'error' else 0]
    return lines, labs

def csv_fmt(rng):
    lines = ['id,name,score', '# exported 2026-11-04']; labs = [0, 0]
    for i in range(rng.randint(4, 8)):
        lines += [f'{i},{rng.choice(W)},{rng.random():.3f}']; labs += [1]
    return lines, labs

def yaml_cfg(rng):
    lines = ['# auto-generated, do not edit', '---', f'version: "{rng.randint(1,4)}"']; labs = [0, 0, 0]
    for _ in range(rng.randint(3, 6)):
        lines += [f'{rng.choice(W)}: {rng.choice(W)}-{rng.randint(1,99)}  # {rng.choice(CONTENT)[:40]}']
        labs += [1]
    return lines, labs

def prose_para(rng):
    lines = ['Click here to subscribe to our newsletter.', 'All rights reserved 2026.']; labs = [0, 0]
    for _ in range(rng.randint(3, 6)):
        lines += [rng.choice(CONTENT)]; labs += [1]
    lines += ['Share on Twitter Facebook LinkedIn.']; labs += [0]
    return lines, labs

FORMATS = {"json": json_api, "xml": xml_feed, "markdown": markdown_doc,
           "logfmt": logfmt_events, "csv": csv_fmt, "yaml": yaml_cfg, "prose": prose_para}

def make(seed=0, per_format=150):
    rng = random.Random(seed)
    out = {}
    for name, fn in FORMATS.items():
        fam = []
        for _ in range(per_format):
            lines, labs = fn(rng)
            assert len(lines) == len(labs), (name, len(lines), len(labs))
            fam.append({"lines": lines, "labels": labs})
        out[name] = fam
    return out

if __name__ == "__main__":
    c = make()
    n = sum(len(s["lines"]) for f in c.values() for s in f)
    k = sum(sum(s["labels"]) for f in c.values() for s in f)
    print(f"formats={len(c)} lines={n} keep_rate={k/n:.2f}")
