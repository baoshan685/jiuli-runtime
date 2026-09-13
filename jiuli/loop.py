# -*- coding: utf-8 -*-
"""运行循环：把检索、组装、生成、解析、校验、存档串成一个回合。

每回合 = 一次 LLM 调用。事实记账（状态/存档/记忆）全部在代码侧完成。
"""
import json
from pathlib import Path

from jiuli.context import ContextAssembler, est_tokens
from jiuli.memory import MemoryStore, heuristic_extract_facts
from jiuli.parser import extract_statusbar, parse_output
from jiuli.retriever import WorldbookRetriever
from jiuli.session import SessionStore
from jiuli.state import StateMachine

DEFAULT_INSTRUCTION = (
    "你是沉浸式虚构故事游戏的叙述引擎。严格依据【人设】与【设定】扮演角色，"
    "禁止代替玩家行动或描写玩家内心。"
    "每次回复都必须以一个尾部块结束（必须存在，不能省略）：```jiuli {json} ```，"
    "json 字段包含 state_diff（状态变化，如 {\"affection\": \"+1\"}，"
    "以【当前状态】中存在的变量为准）、"
    "memory_facts（本回合**实际发生**的、值得角色长期记住的事实数组——"
    "只记录本回合发生的事，禁止虚构此前未发生的剧情；没有则空数组）、"
    "suggestions（3 个剧情建议）。尾部块之外只输出叙事正文。"
    "若【设定】中包含状态栏格式要求（如 <userStatusBlocks> 等标签包裹的块），"
    "则每一轮都【必须】在叙事正文之后、尾部块之前，严格按该格式输出完整状态栏块，"
    "不得省略。运行时会渲染它。"
)


class CardPackage:
    def __init__(self, pkg_dir):
        pkg = Path(pkg_dir)
        self.dir = pkg
        self.manifest = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
        persona = json.loads((pkg / "persona.json").read_text(encoding="utf-8"))
        self.persona = persona
        self.persona_text = "名字：{name}。{bio} 场景：{scenario}".format(
            name=persona.get("name", ""), bio=persona.get("bio", ""),
            scenario=persona.get("scenario", "") or "（见世界书）")
        sm = json.loads((pkg / "state_machine.json").read_text(encoding="utf-8"))
        self.state_machine = StateMachine(sm)
        self.retriever = WorldbookRetriever(pkg)
        # 常驻条目：等同预设，每轮固定注入（P0 实验原则的"始终注入"半边，
        # 此前只有"不参与检索"半边——状态栏指令/常驻世界规则因此丢失）
        self.constant_entries = [
            (e["id"], e.get("content", ""))
            for e in self.retriever_entries_all()
        ]

    def retriever_entries_all(self):
        """全部条目（含常驻），从 index+文件重建。"""
        import json as _json
        index = _json.loads((self.dir / "worldbook" / "index.json").read_text(encoding="utf-8"))
        out = []
        for e in index["entries"]:
            if e["constant"] and e["enabled"]:
                content = (self.dir / "worldbook" / e["content_file"]).read_text(encoding="utf-8")
                out.append(dict(e, content=content))
        return out


def _make_nli(llm):
    """LLM 语义矛盾裁判：True=推翻 False=不矛盾 None=弃权。"""
    def check(new_text, old_text):
        raw = llm.chat(
            "判断：第二条事实是否在推翻或矛盾于第一条事实？只回答 yes 或 no。",
            [{"role": "user", "content": "1) %s\n2) %s" % (old_text, new_text)}],
            temperature=0.0, max_tokens=512)
        low = (raw or "").strip().lower()
        if "yes" in low:
            return True
        if "no" in low:
            return False
        return None
    return check


class RPSession:
    def __init__(self, pkg_dir, llm, db_path, session_id=None,
                 token_budget=4000, fact_extractor=None, card_id=""):
        self.pkg = CardPackage(pkg_dir)
        self.llm = llm
        self.store = SessionStore(db_path)
        # 真实 LLM 时启用语义矛盾裁判；MockLLM 退回纯规则
        nli = None if type(self.llm).__name__ == "MockLLM" else _make_nli(llm)
        self.memory = MemoryStore(self.store, nli_check=nli)
        self.assembler = ContextAssembler(
            persona=self.pkg.persona_text,
            state_line_fn=lambda: json.dumps(self.state, ensure_ascii=False),
            instruction=DEFAULT_INSTRUCTION,
            token_budget=token_budget)
        self.extractor = fact_extractor or heuristic_extract_facts
        if session_id is None:
            self.session_id = self.store.create_session(
                title=self.pkg.persona.get("name", ""), card_id=card_id)
            self.store.set_state(self.session_id, self.pkg.state_machine.initial_state())
        else:
            self.session_id = session_id
        self._state = self.store.get_state(self.session_id)
        self.last_warnings = []
        self._last_added_fact_ids = []  # 本会话累计新增的事实 id，供 reroll 撤回

    @property
    def state(self):
        return self._state

    def run_turn(self, user_input, record=True):
        return self._turn(user_input, record=record)

    def reroll(self):
        """swipe：删掉最后一组交换并重新生成。

        被删回合写入的记忆事实一并撤回（含恢复被其推翻的旧事实），
        避免"roll 掉的剧情角色却记得"（P4 复盘缺陷 #4）。
        """
        last_user = self.store.delete_last_exchange(self.session_id)
        if last_user is None:
            return None
        if self._last_added_fact_ids:
            self.store.retract_facts(self._last_added_fact_ids)
            self._last_added_fact_ids = []
            self.memory._dirty = True
        return self._turn(last_user, record=True)

    def fork(self, title=""):
        return self.store.fork_session(self.session_id, title=title)

    def _turn(self, user_input, record=True):
        system, prep = self._prepare(user_input)
        self.last_warnings = []
        raw = self.llm.chat(system, [{"role": "user", "content": user_input}])
        narrative, tail, warns = parse_output(raw)
        if not narrative.strip():
            raw = self.llm.chat(system, [{"role": "user", "content": user_input},
                {"role": "assistant", "content": raw[-400:]},
                {"role": "user", "content": "你的回复缺少叙事正文。请重新输出："
                 "只输出叙事正文，尾部块保持不变。"}], max_tokens=6000)
            narrative, tail, warns = parse_output(raw)
            if not narrative.strip():
                self.last_warnings.append("模型两次返回空正文")
        self.last_warnings.extend(warns)
        result = self._finalize(user_input, narrative, tail, prep, record=record)
        result["ctx_tokens_est"] = est_tokens(system)
        return result

    def run_turn_stream(self, user_input, record=True):
        """流式回合：yield 事件 dict，最后一项为 {"type":"done","result":...}。

        事件：status(检索完成) → ctx(上下文规模) → delta*(正文增量，尾部块不外发)
        → done(与 run_turn 相同的完整结果)。生成失败同样重试一次。
        """
        system, prep = self._prepare(user_input)
        self.last_warnings = []
        yield {"type": "status", "wb_hits": prep["wb_hits"], "mem_hits": len(prep["mem"])}
        yield {"type": "ctx", "tokens": est_tokens(system)}

        def gen(max_tokens=4000):
            if hasattr(self.llm, "chat_stream"):
                for piece in self.llm.chat_stream(
                        system, [{"role": "user", "content": user_input}],
                        max_tokens=max_tokens):
                    yield piece
            else:
                yield self.llm.chat(system, [{"role": "user", "content": user_input}],
                                    max_tokens=max_tokens)

        raw_acc = []
        emitted = 0
        got_any = False
        for piece in gen():
            raw_acc.append(piece)
            raw = "".join(raw_acc)
            tail_at = raw.find("```jiuli")
            visible = raw if tail_at < 0 else raw[:tail_at]
            if len(visible) > emitted:
                got_any = True
                yield {"type": "delta", "text": visible[emitted:]}
                emitted = len(visible)
        if not got_any:
            # 空正文重试（不流式，直接整段）
            raw = self.llm.chat(system, [{"role": "user", "content": user_input}],
                                max_tokens=6000)
            narrative, tail, warns = parse_output(raw)
            if narrative.strip():
                yield {"type": "delta", "text": narrative}
            else:
                self.last_warnings.append("模型两次返回空正文")
            self.last_warnings.extend(warns)
            yield {"type": "done", "result": self._finalize(
                user_input, narrative, tail, prep, record=record)}
            return
        raw = "".join(raw_acc)
        narrative, tail, warns = parse_output(raw)
        self.last_warnings.extend(warns)
        result = self._finalize(user_input, narrative, tail, prep, record=record)
        yield {"type": "done", "result": result}

    def _prepare(self, user_input):
        """检索 + 组装，供非流式/流式共用。返回 (system, prep)。"""
        history = self.store.recent_messages(self.session_id)
        hits = self.pkg.retriever.search(user_input, topk=6)
        # 常驻条目置顶（高分保证排最前），检索命中随后
        wb = ([(eid, 9999.0, content) for eid, content in self.pkg.constant_entries]
              + [(eid, s, (self.pkg.retriever.get(eid) or {}).get("content", ""))
                 for eid, s in hits])
        mem = self.memory.search(user_input, topk=4) or self.memory.recent(3)
        total = self.store.count_messages(self.session_id)
        older = max(0, total - len(history))
        system, dropped = self.assembler.assemble(
            history, wb, mem,
            summary=self.store.get_summary(self.session_id), older_count=older)
        prep = {"hits": hits, "wb_hits": [eid for eid, _ in hits], "mem": mem,
                "dropped": dropped}
        return system, prep

    def _finalize(self, user_input, narrative, tail, prep, record=True):
        applied, rejected, illustration, suggestions = {}, [], None, []
        if tail:
            if "state_diff" in tail:
                self._state, applied, rejected = self.pkg.state_machine.validate_and_apply(
                    self._state, tail["state_diff"])
                if rejected:
                    self.last_warnings.append("状态 diff 部分被拒: %s" % rejected)
            illustration = tail.get("illustration")
            suggestions = tail.get("suggestions", []) or []
            if tail.get("summary"):
                self.store.set_summary(self.session_id, str(tail["summary"]))

        if record:
            self.store.add_message(self.session_id, "user", user_input)
            self.store.add_message(self.session_id, "assistant", narrative)
        self.store.set_state(self.session_id, self._state)
        facts = tail.get("memory_facts", []) if tail else []
        facts = facts or self.extractor(user_input + "\n" + narrative)
        if facts:
            stats = self.memory.add(self.session_id, facts, context=user_input)
            self._last_added_fact_ids.extend(stats.get("added_ids", []))

        narrative, statusbar = extract_statusbar(narrative)
        return {
            "narrative": narrative,
            "statusbar": statusbar,
            "state": dict(self._state),
            "applied": applied,
            "rejected": rejected,
            "illustration": illustration,
            "suggestions": suggestions,
            "warnings": self.last_warnings,
            "wb_hits": prep["wb_hits"],
            "mem_hits": len(prep["mem"]),
            "ctx_tokens_est": 0,  # 由调用方覆盖；非流式路径使用 prepare 结果
            "history_dropped": prep["dropped"],
            "usage": getattr(self.llm, "last_usage", None),
        }
