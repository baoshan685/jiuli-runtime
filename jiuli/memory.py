# -*- coding: utf-8 -*-
"""记忆引擎 v1：去重、纠错消解、遗忘修剪、来源标记。

原则（P4 设计）：
- 事实是结构化记录：{text, kind}，kind ∈ event/relationship/world/preference
- 只记录"本回合发生的事"（提示词收紧），模型虚构的前史靠来源标记暴露
- 近重复（bigram Jaccard ≥ 0.7）不重复入库
- 纠错消解：新事实带纠正线索（其实/不是/误会/记错/改为）且与旧事实词面重叠
  时，旧事实停用（retcon），新事实继承加权
- 遗忘修剪：生效事实超过上限时从最旧开始停用
检索复用世界书同款字符 bigram BM25。
"""
from jiuli.retriever import BM25, tokenize

FACT_KINDS = ("event", "relationship", "world", "preference")

CORRECTION_CUES = ("其实", "不是", "误会", "记错", "改为", "取消", "并没有",
                   "其实不是", "错怪", "推翻")

# 首轮成对矛盾词表（保守版：仅同向反转，避免误杀）
CONTRADICT_PAIRS = [
    ("喜欢", "讨厌"), ("讨厌", "喜欢"),
    ("原谅", "记恨"), ("记恨", "原谅"),
    ("在一起", "分手"), ("分手", "复合"),
    ("答应了", "拒绝了"), ("拒绝了", "答应了"),
]


def _bigrams(text):
    return set(tokenize(text))


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _overlap(a, b):
    """重叠系数：|A∩B| / min(|A|,|B|)。对长短句差异远比 Jaccard 公平，
    适合"短纠正事实 vs 长原始事实"的消解场景。"""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def normalize_facts(facts):
    """接受 str / {text, kind} 混合列表，规整为 dict 列表。"""
    out = []
    for f in facts or []:
        if isinstance(f, str):
            f = {"text": f, "kind": "event"}
        t = (f.get("text") or "").strip()
        if t:
            out.append({"text": t, "kind": f.get("kind", "event")})
    return out


def _has_cue(new_text, context=None):
    return any(c in new_text for c in CORRECTION_CUES) or (
        bool(context) and any(c in context for c in CORRECTION_CUES))


def find_contradiction(new_text, existing, context=None):
    """返回应被停用的旧事实 id 列表。existing: [dict(id,text,...)]。

    v0 规则（线索来源 = 新事实文本 或 产生它的玩家原话 context）：
    1) 纠正线索 + 重叠系数 ≥ 0.4 → 旧事实让位
    2) 成对矛盾词命中同一旧事实且重叠系数 ≥ 0.35 → 旧事实让位
    都不命中则不做消解（NLI 裁判由 MemoryStore 在此之后按需调用）。
    """
    new_bg = _bigrams(new_text)
    has_cue = _has_cue(new_text, context)
    supersede = []
    for old in existing:
        ov = _overlap(new_bg, _bigrams(old["text"]))
        if has_cue and ov >= 0.4:
            supersede.append(old["id"])
            continue
        for w1, w2 in CONTRADICT_PAIRS:
            if w1 in new_text and w2 in old["text"] and ov >= 0.35:
                supersede.append(old["id"])
                break
    return supersede


class MemoryStore:
    def __init__(self, store, max_facts=500, nli_check=None):
        """store: SessionStore（facts 表）。max_facts: 遗忘修剪上限。

        nli_check: 可选 (new_text, old_text) -> bool|None 的语义矛盾裁判
        （LLM 实现）。规则消解不命中但存在纠正线索时调用；None/异常 视为
        弃权，维持规则结果。无 LLM 时保持 None，纯规则兜底。
        """
        self.store = store
        self.max_facts = max_facts
        self.nli_check = nli_check
        self._dirty = True
        self._bm25 = None
        self._records = []
        self._version = None

    # ── 写入 ──────────────────────────────────────────────────
    def add(self, session_id, facts, context=None):
        """facts: str/dict 混合列表；context: 产生这些事实的玩家原话（供纠错线索）。"""
        added, dupes, superseded = 0, 0, 0
        added_ids = []
        for f in normalize_facts(facts):
            existing = self._refresh()
            new_bg = _bigrams(f["text"])
            if any(_jaccard(new_bg, _bigrams(e["text"])) >= 0.7 for e in existing):
                dupes += 1
                continue
            victims = find_contradiction(f["text"], existing, context=context)
            if not victims and self.nli_check and (
                    _has_cue(f["text"], context)):
                # 词面规则到顶时的语义裁判：按重叠系数取最接近的 2 条候选
                cands = sorted(existing, key=lambda e: -_overlap(
                    new_bg, _bigrams(e["text"])))[:2]
                for old in cands:
                    try:
                        if self.nli_check(f["text"], old["text"]) is True:
                            victims.append(old["id"])
                    except Exception:  # noqa: BLE001
                        continue
            fid = self.store.add_fact(session_id, f["text"], kind=f["kind"])
            added_ids.append(fid)
            for vid in victims:
                self.store.deactivate_fact(vid, superseded_by=fid)
                superseded += 1
            added += 1
            existing.append(dict(id=fid, text=f["text"], kind=f["kind"]))
            if len(existing) > self.max_facts:
                self.store.prune_facts(self.max_facts)
                self._dirty = True
                existing = self._refresh()
            else:
                self._records = existing
        if added:
            self._dirty = True
        return {"added": added, "duplicates": dupes, "superseded": superseded,
                "added_ids": added_ids}

    # ── 检索 ──────────────────────────────────────────────────
    def _refresh(self):
        # 多会话共享 facts 表：其他会话写入后本缓存必须失效（P4 复盘缺陷 #2）。
        # 轻量版本号检查避免"另一个会话刚发生的事，这个会话想不起来"。
        version = self.store.facts_version()
        if self._bm25 is not None and not self._dirty and version != self._version:
            self._dirty = True
        if not self._dirty:
            return self._records
        self._records = self.store.active_facts()
        self._bm25 = BM25([tokenize(r["text"]) for r in self._records]) \
            if self._records else None
        self._version = version
        self._dirty = False
        return self._records

    def search(self, query, topk=5, kinds=None):
        self._refresh()
        if not self._bm25:
            return []
        qt = tokenize(query)
        scored = sorted(((self._bm25.score(qt, i), i, r)
                         for i, r in enumerate(self._records)),
                        key=lambda x: (-x[0], x[1]))
        out = []
        for s, _, r in scored:
            if s > 0 and (kinds is None or r["kind"] in kinds):
                out.append(r["text"])
            if len(out) >= topk:
                break
        return out

    def recent(self, topk=3):
        """最近的若干条事实——查询式检索无命中时的兜底注入。"""
        self._refresh()
        return [r["text"] for r in self._records[-topk:]]


# ── LLM 事实抽取（v1：结构化 kind + 收紧到"只记录本回合"） ──────────

FACT_EXTRACT_PROMPT = """从以下角色扮演对话片段中，抽取 0-5 条「角色会长期记住的事实」。
要求：
- 只记录【本回合实际发生】的事件、承诺、关系变化；禁止推测或虚构此前未发生的剧情
- 以角色视角/第三人称陈述，如「玩家在雨天借给过她一把伞」
- 每条标注 kind，取值: event(事件)/relationship(关系变化)/world(世界观变动)/preference(偏好)
- 输出 JSON 数组，每项 {{"text": "...", "kind": "..."}}

对话片段：
{chunk}
"""


def heuristic_extract_facts(chunk):
    """无 LLM 时的兜底抽取：返回空（宁缺勿滥，避免噪声入库）。"""
    return []
