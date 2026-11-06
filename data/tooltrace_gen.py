"""Synthetic agent tool-trace generator with ground-truth line labels.

Labels: 1 = keep (information the agent needs), 0 = drop (boilerplate/noise).

Design principle: each family mimics a real tool output an LLM agent sees.
Content text is drawn from a pool of plausible sentences; boilerplate from
real-world templates (pip, HTTP, docker, pytest...). Families are the unit
of the train/test split, so we measure generalization to UNSEEN tools.
"""
import random, json

CONTENT_SENTENCES = [
    "The quarterly revenue grew 14% year over year to 2.3 million dollars.",
    "Migration 0042_alter_users_table applied successfully in 0.83 seconds.",
    "User sarah.connor@example.com last signed in on 2026-10-30 from Vancouver.",
    "The experiment shows a 7.1 point F1 improvement over the baseline model.",
    "Endpoint /v2/orders accepts POST with fields item_id, quantity and note.",
    "The committee approved the budget after a two hour debate on Tuesday.",
    "Function retry_with_backoff caps at 8 attempts with jitter enabled.",
    "Average response latency dropped from 480ms to 210ms after the cache fix.",
    "The novel follows a lighthouse keeper on the coast of Newfoundland.",
    "Researchers found the compound reduced plaque volume by 32 percent.",
    "Set MAX_CONNECTIONS=64 in production to avoid pool exhaustion.",
    "The release includes security fixes for CVE-2026-1189 and CVE-2026-2044.",
    "Battery capacity is 4800 mAh with 65W fast charging support.",
    "She earned her doctorate in linguistics from the University of Toronto.",
    "The play opens in a crowded market square at dawn.",
    "Our analysis indicates the outage was caused by a misconfigured router.",
]

PKGS = ["numpy", "requests", "flask", "django", "torch", "pandas", "scipy", "redis", "boto3", "click"]
HOSTS = ["api.example.com", "data.acme.io", "internal.corp.net", "cdn.static.org"]
WORDS = "alpha beta gamma delta service worker cache node cluster proxy shard queue".split()

def _content(rng, n):
    return [rng.choice(CONTENT_SENTENCES) for _ in range(n)]

def _bar(pct): return f"[{'=' * (pct // 5):<20}] {pct:3d}%"

def pip_install(rng):
    lines, labs = [], []
    pkg, ver = rng.choice(PKGS), f"{rng.randint(0,9)}.{rng.randint(0,20)}.{rng.randint(0,9)}"
    lines += [f"Collecting {pkg}"]; labs += [0]
    for _ in range(rng.randint(2, 5)):
        lines += [f"  Downloading {pkg}-{ver}-cp310-manylinux.whl ({rng.randint(1,40)}.{rng.randint(0,9)} MB)"]
        labs += [0]
        for p in range(0, 101, rng.choice([10, 20, 25])):
            lines += [f"     {_bar(p)} {rng.randint(1,99)}.{rng.randint(0,9)} MB/s eta 0:00:0{rng.randint(0,9)}"]; labs += [0]
    for dep in rng.sample(PKGS, rng.randint(1, 4)):
        lines += [f"Requirement already satisfied: {dep} in ./venv/lib/python3.10/site-packages"]; labs += [0]
    lines += ["Installing collected packages: " + pkg, "Successfully installed " + pkg + "-" + ver]; labs += [0, 1]
    return lines, labs

def http_api(rng):
    lines = [f"GET https://{rng.choice(HOSTS)}/v1/items HTTP/1.1", "HTTP/1.1 200 OK",
             "Date: Tue, 04 Nov 2026 1{0}:22:31 GMT".format(rng.randint(0,9)),
             "Content-Type: application/json; charset=utf-8",
             f"X-Request-Id: {rng.randrange(16**8):08x}-{rng.randrange(16**4):04x}",
             "Cache-Control: no-store", ""]
    labs = [0] * len(lines)
    items = [{"id": rng.randint(1000, 9999), "fact": rng.choice(CONTENT_SENTENCES)} for _ in range(rng.randint(2, 5))]
    body = json.dumps({"items": items, "page": 1}, indent=2).splitlines()
    for b in body:
        lines.append(b); labs.append(1 if "fact" in b else 0)
    return lines, labs

def git_log(rng):
    lines, labs = [], []
    for _ in range(rng.randint(3, 6)):
        lines += [f"commit {rng.randrange(16**40):040x}", f"Author: Dev {rng.choice(WORDS).title()} <dev@example.com>",
                  f"Date:   Wed Oct {rng.randint(1,31)} 2026 -0700", ""]
        labs += [0, 0, 0, 0]
        msg = rng.choice(CONTENT_SENTENCES)
        lines += [f"    {msg}"]; labs += [1]
        if rng.random() < 0.4:
            lines += ["    " + rng.choice(CONTENT_SENTENCES)]; labs += [1]
    return lines, labs

def search_results(rng):
    lines = ['{', '  "query": "internal search",', f'  "total": {rng.randint(100,9999)},', '  "results": [']
    labs = [0, 0, 0, 0]
    for i in range(rng.randint(3, 6)):
        lines += ['    {', f'      "title": "Result {i} - {rng.choice(WORDS)} docs",',
                  f'      "url": "https://{rng.choice(HOSTS)}/docs/{i}",',
                  f'      "snippet": "{rng.choice(CONTENT_SENTENCES)}"', '    },']
        labs += [0, 1, 1, 1, 0]
    lines += ['  ],', f'  "took_ms": {rng.randint(3, 99)}', '}']; labs += [0, 0, 0]
    return lines, labs

def pytest_out(rng):
    lines, labs = ["============================= test session starts =============================",
                   f"platform linux -- Python 3.10.{rng.randint(0,9)}, pytest-{rng.randint(6,8)}.{rng.randint(0,9)}.0",
                   "rootdir: /home/dev/project", "collected 48 items"], [0, 0, 0, 0]
    lines += ["tests/test_api.py " + "." * rng.randint(5, 20)]; labs += [0]
    lines += ["tests/test_core.py " + "." * rng.randint(5, 12) + "F" + "." * rng.randint(2, 8)]; labs += [0]
    lines += ["=================================== FAILURES ===================================",
              "_______________________________ test_checkout ________________________________",
              "E       AssertionError: " + rng.choice(CONTENT_SENTENCES)]
    labs += [0, 0, 1]
    lines += [f"========================= 1 failed, 47 passed in {rng.randint(3,30)}.{rng.randint(0,9)}s ========================="]
    labs += [1]
    return lines, labs

def docker_build(rng):
    lines, labs = [], []
    for i in range(rng.randint(4, 9)):
        lines += [f"Step {i}/12 : RUN apt-get install -y {rng.choice(PKGS)}", " ---> Using cache",
                  f" ---> {rng.randrange(16**12):012x}"]
        labs += [0, 0, 0]
    if rng.random() < 0.5:
        lines += ["E: Unable to locate package " + rng.choice(PKGS),
                  "The command '/bin/sh -c apt-get install' returned a non-zero code: 100"]; labs += [1, 1]
    else:
        lines += ["Successfully built " + f"{rng.randrange(16**12):012x}"]; labs += [1]
    return lines, labs

def stack_trace(rng):
    lines = ["Traceback (most recent call last):"]; labs = [0]
    for i in range(rng.randint(3, 7)):
        lines += [f'  File "/app/{rng.choice(WORDS)}/{rng.choice(WORDS)}.py", line {rng.randint(1,400)}, in {rng.choice(WORDS)}_handler',
                  f"    return self.{rng.choice(WORDS)}(request, *args)"]
        labs += [0, 0]
    lines += [f"ValueError: {rng.choice(CONTENT_SENTENCES)}"]; labs += [1]
    return lines, labs

def html_dump(rng):
    lines = ["<nav>", "  Home | Products | About | Contact | Sign in", "</nav>",
             "<div class=\"cookie-banner\">", "  We use cookies to improve your experience. Accept all?",
             "</div>", "<article>"]
    labs = [0] * 7
    for _ in range(rng.randint(3, 6)):
        lines += [f"  <p>{rng.choice(CONTENT_SENTENCES)}</p>"]; labs += [1]
    lines += ["</article>", "<footer>", "  Copyright 2026 Example Corp. All rights reserved.",
              "  Privacy Policy | Terms of Service", "</footer>"]
    labs += [0] * 5
    return lines, labs

def kubectl_out(rng):
    lines = ["NAME                          READY   STATUS    RESTARTS   AGE"]; labs = [0]
    for _ in range(rng.randint(3, 7)):
        lines += [f"{rng.choice(WORDS)}-{rng.randrange(16**5):05x}   {rng.randint(0,1)}/1     Running   {rng.randint(0,9)}          {rng.randint(1,40)}d"]
        labs += [1]
    return lines, labs

def csv_table(rng):
    lines = ["id,name,score,region", "# exported 2026-11-04, encoding utf-8", "# do not edit"]; labs = [0, 0, 0]
    for i in range(rng.randint(4, 9)):
        lines += [f"{i},{rng.choice(WORDS)},{rng.random():.3f},{rng.choice(['us','eu','ap'])}"]; labs += [1]
    return lines, labs

FAMILIES = {"pip": pip_install, "http": http_api, "git": git_log, "search": search_results,
            "pytest": pytest_out, "docker": docker_build, "stack": stack_trace,
            "html": html_dump, "kubectl": kubectl_out, "csv": csv_table}

def make_corpus(seed=0, per_family=120):
    rng = random.Random(seed)
    corpus = {}
    for name, fn in FAMILIES.items():
        fam = []
        for _ in range(per_family):
            lines, labs = fn(rng)
            fam.append({"lines": lines, "labels": labs})
        corpus[name] = fam
    return corpus

if __name__ == "__main__":
    import json as j
    c = make_corpus()
    n_lines = sum(len(s["lines"]) for f in c.values() for s in f)
    n_keep = sum(sum(s["labels"]) for f in c.values() for s in f)
    print(f"families={len(c)} traces={sum(len(f) for f in c.values())} lines={n_lines} keep_rate={n_keep/n_lines:.2f}")
