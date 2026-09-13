# -*- coding: utf-8 -*-
"""世界书检索：key（酒馆式字面触发）+ lex（字符 bigram BM25）+ hybrid（RRF）。

算法与 P0 实验验证的版本一致（p0-retrieval-exp/REPORT.md）：
字面型查询 R@5=1.000，中文改写型 R@5=0.826。零第三方依赖。
检索候选 = 非常驻条目；常驻条目始终注入、不参与检索。
"""
import json
import math
import re
from pathlib import Path


def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9._]*", text)
    cjk = re.sub(r"[^\u4e00-\u9fff\u3040-\u30ff]", "",
                 re.sub(r"[a-z0-9._\s]+", "", text))
    for i in range(len(cjk) - 1):
        tokens.append(cjk[i:i + 2])
    return tokens


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.avgdl = sum(len(d) for d in docs_tokens) / max(1, self.N)
        self.tf = [{} for _ in docs_tokens]
        self.df = {}
        for i, d in enumerate(docs_tokens):
            for t in d:
                self.tf[i][t] = self.tf[i].get(t, 0) + 1
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1

    def idf(self, t):
        n = self.df.get(t, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1)

    def score(self, query_tokens, i):
        s = 0.0
        for t in query_tokens:
            f = self.tf[i].get(t, 0)
            if f:
                dl = len(self.tf[i])
                s += self.idf(t) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


class WorldbookRetriever:
    """从卡片语义包（worldbook/index.json + entries/*.md）构建检索器。"""

    def __init__(self, pkg_dir):
        pkg = Path(pkg_dir)
        index = json.loads((pkg / "worldbook" / "index.json").read_text(encoding="utf-8"))
        self.entries = []
        for e in index["entries"]:
            if e["constant"] or not e["enabled"]:
                continue
            content = (pkg / "worldbook" / e["content_file"]).read_text(encoding="utf-8")
            self.entries.append(dict(e, content=content))
        self._bm25 = None
        if self.entries:
            docs = []
            for e in self.entries:
                text = (e["title"] + " ") * 2 + (" ".join(e["keys"]) + " ") * 3 + e["content"]
                docs.append(tokenize(text))
            self._bm25 = BM25(docs)

    def search_key(self, query, topk=8):
        hits, q = [], query.lower()
        for e in self.entries:
            matched = [k for k in e["keys"] if k.lower() in q]
            if matched:
                hits.append((e["id"], len(matched) + 0.1 * sum(len(k) for k in matched)))
        hits.sort(key=lambda x: -x[1])
        return hits[:topk]

    def search_lex(self, query, topk=8):
        if not self._bm25:
            return []
        qt = tokenize(query)
        scored = sorted(((self._bm25.score(qt, i), e["id"])
                         for i, e in enumerate(self.entries)), reverse=True)
        return [(eid, s) for s, eid in scored if s > 0][:topk]

    def search(self, query, topk=8, k=60):
        """hybrid RRF——运行时默认通道。"""
        r1 = self.search_key(query, topk=topk)
        r2 = self.search_lex(query, topk=topk)
        rrf = {}
        for rank, (eid, _) in enumerate(r1):
            rrf[eid] = rrf.get(eid, 0) + 1 / (k + rank + 1)
        for rank, (eid, _) in enumerate(r2):
            rrf[eid] = rrf.get(eid, 0) + 1 / (k + rank + 1)
        return sorted(rrf.items(), key=lambda x: -x[1])[:topk]

    def get(self, entry_id):
        for e in self.entries:
            if e["id"] == entry_id:
                return e
        return None
