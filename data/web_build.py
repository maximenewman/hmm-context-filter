"""Real web-page corpus with DOM-derived silver labels.

Fetches pages, parses with stdlib html.parser. Text inside
nav/footer/header/aside/script/style/form/noscript/banner = drop.
Text inside article/main, or <p>/<h1-6>/<li>/<pre> outside boiler tags = keep.
Everything else = drop (silver standard: most other text is chrome).
The label comes from document STRUCTURE, not from text features,
so it is an honest external label for the filter to predict.
"""
import json, re, sys, urllib.request
from html.parser import HTMLParser

DROP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "form",
             "noscript", "select", "button", "iframe", "svg"}
KEEP_CONTEXT = {"article", "main"}
KEEP_INLINE = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "blockquote"}

class BlockParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.blocks = [], []

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        text = re.sub(r"\s+", " ", data).strip()
        if len(text) < 3:
            return
        in_drop = any(t in DROP_TAGS for t in self.stack)
        in_keep_ctx = any(t in KEEP_CONTEXT for t in self.stack)
        inner = self.stack[-1] if self.stack else ""
        keep = (not in_drop) and (in_keep_ctx or inner in KEEP_INLINE)
        self.blocks.append({"text": text, "keep": 0 if in_drop else (1 if keep else 0),
                            "tag": inner})

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research-corpus)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    enc = "utf-8"
    m = re.search(br'charset=["\']?([\w-]+)', raw[:4000])
    if m:
        enc = m.group(1).decode("ascii", "ignore") or "utf-8"
    return raw.decode(enc, "replace")

def build(urls, out_path):
    corpus = []
    for url in urls:
        try:
            html = fetch(url)
        except Exception as e:
            print(f"SKIP {url}: {e}", file=sys.stderr)
            continue
        p = BlockParser()
        p.feed(html)
        blocks = p.blocks
        keeps = sum(b["keep"] for b in blocks)
        if keeps < 5 or len(blocks) < 20:
            print(f"SKIP {url}: too few blocks ({len(blocks)}, keep={keeps})", file=sys.stderr)
            continue
        corpus.append({"url": url, "lines": [b["text"] for b in blocks],
                       "labels": [b["keep"] for b in blocks]})
        print(f"OK {url}: {len(blocks)} blocks, keep={keeps}", file=sys.stderr)
    with open(out_path, "w") as f:
        json.dump(corpus, f)
    return corpus

URLS = [
    "https://en.wikipedia.org/wiki/Hidden_Markov_model",
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://en.wikipedia.org/wiki/Vancouver",
    "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
    "https://docs.python.org/3/tutorial/introduction.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/library/json.html",
    "https://docs.python.org/3/howto/logging.html",
    "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status",
    "https://developer.mozilla.org/en-US/docs/Learn_web_development/Getting_started/Your_first_website",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Loops_and_iteration",
    "https://arstechnica.com/information-technology/",
    "https://www.bbc.com/news/technology",
    "https://www.bbc.com/sport",
    "https://blog.rust-lang.org/",
    "https://numpy.org/doc/stable/user/quickstart.html",
    "https://scikit-learn.org/stable/getting_started.html",
    "https://pytorch.org/get-started/locally/",
    "https://en.wikipedia.org/wiki/Context-free_grammar",
    "https://en.wikipedia.org/wiki/Information_theory",
    "https://en.wikipedia.org/wiki/Simon_Fraser_University",
    "https://docs.github.com/en/actions/quickstart",
    "https://kubernetes.io/docs/concepts/overview/",
    "https://en.wikipedia.org/wiki/Compression_(physics)",
    "https://en.wikipedia.org/wiki/Data_compression",
    "https://www.nps.gov/yell/learn/nature/geology.htm",
    "https://www.canada.ca/en/services/health.html",
    "https://en.wikipedia.org/wiki/Bayes%27_theorem",
    "https://en.wikipedia.org/wiki/Markov_chain",
]

if __name__ == "__main__":
    c = build(URLS, "data/web_corpus.json")
    tot = sum(len(x["lines"]) for x in c)
    keep = sum(sum(x["labels"]) for x in c)
    print(f"pages={len(c)} blocks={tot} keep_rate={keep/max(tot,1):.2f}")
