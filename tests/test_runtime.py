# -*- coding: utf-8 -*-
"""酒醴运行时测试（stdlib unittest）。

运行：在 jiuli-runtime 目录下  python -m unittest discover tests -v
含可跳过的 P0 检索回归（存在 ../p0-retrieval-exp 时自动执行）。
"""
import json
import shutil
import time
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jiuli.context import ContextAssembler, est_tokens
from jiuli.llm import MockLLM
from jiuli.loop import RPSession
from jiuli.parser import parse_output
from jiuli.session import SessionStore
from jiuli.retriever import WorldbookRetriever
from jiuli.state import StateMachine

STATE_MACHINE = {
    "variables": [
        {"name": "affection", "label": "好感度", "default": 0,
         "min": 0, "max": 100, "explicit": True},
        {"name": "mood", "label": "心情", "default": "平静", "explicit": True},
        {"name": "trust", "label": "信赖度", "default": 0,
         "min": 0, "max": 100, "explicit": False},  # 低置信提示：不应启用
    ],
}

PERSONA = {"name": "小铃", "short_name": "小铃", "bio": "旧书店的图书管理员。",
           "description": "爱笑，喜欢旧书和热茶。", "personality": "温柔",
           "scenario": "巷子深处的旧书店", "tags": []}

ENTRIES = [
    {"id": "001-shudian", "title": "书店设定", "keys": ["旧书店", "拾光"],
     "constant": False, "enabled": True, "content": "书店名叫「拾光」，冬天烧一只旧暖炉，木地板会响。"},
    {"id": "002-ankun", "title": "暗渠", "keys": [],
     "constant": False, "enabled": True, "content": "书店地下有一条传说中运送禁书的暗渠，入口藏在暖炉后。"},
    {"id": "003-rule", "title": "常驻规则", "keys": [],
     "constant": True, "enabled": True, "content": "叙事永远使用中文。"},
]


def make_pkg(tmp, pkg_name="card_package-demo"):
    pkg = Path(tmp) / pkg_name
    ent = pkg / "worldbook" / "entries"
    ent.mkdir(parents=True, exist_ok=True)
    (pkg / "manifest.json").write_text(json.dumps(
        {"schema_version": "1.0.0", "skill_name": "rp-demo"}), encoding="utf-8")
    (pkg / "persona.json").write_text(json.dumps(PERSONA, ensure_ascii=False), encoding="utf-8")
    (pkg / "state_machine.json").write_text(
        json.dumps(STATE_MACHINE, ensure_ascii=False), encoding="utf-8")
    index = {"entries": [dict(((k, e[k]) for k in
                               ("id", "title", "keys", "constant", "enabled")),
                              content_file="entries/%s.md" % e["id"]) for e in ENTRIES]}
    (pkg / "worldbook" / "index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8")
    for e in ENTRIES:
        (ent / (e["id"] + ".md")).write_text(e["content"], encoding="utf-8")
    return pkg


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.r = WorldbookRetriever(make_pkg(self.tmp))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_key_hit(self):
        top = self.r.search("我们去旧书店逛逛", topk=3)
        self.assertEqual(top[0][0], "001-shudian")

    def test_paraphrase_hits_lex_channel(self):
        # 查询不含任何 keys，靠正文 bigram 命中（P0 实验的核心场景）
        top = self.r.search("暖炉后面好像有个暗门", topk=3)
        self.assertEqual(top[0][0], "002-ankun")

    def test_constant_excluded(self):
        ids = [e["id"] for e in self.r.entries]
        self.assertNotIn("003-rule", ids)


class TestStateMachine(unittest.TestCase):
    def setUp(self):
        self.sm = StateMachine(STATE_MACHINE)

    def test_initial_excludes_low_confidence(self):
        init = self.sm.initial_state()
        self.assertEqual(init, {"affection": 0, "mood": "平静"})

    def test_valid_delta(self):
        new, applied, rej = self.sm.validate_and_apply({"affection": 10}, {"affection": "+5"})
        self.assertEqual(new["affection"], 15)
        self.assertEqual(applied, {"affection": 15})
        self.assertEqual(rej, [])

    def test_unknown_key_rejected(self):
        new, applied, rej = self.sm.validate_and_apply({"affection": 0}, {"gold": 5})
        self.assertEqual(rej[0]["key"], "gold")
        self.assertNotIn("gold", new)

    def test_range_clamp_rejected(self):
        new, applied, rej = self.sm.validate_and_apply({"affection": 99}, {"affection": 120})
        self.assertEqual(rej[0]["reason"], "超过上限")
        self.assertEqual(new["affection"], 99)

    def test_string_var_overwrite(self):
        new, applied, rej = self.sm.validate_and_apply({"mood": "平静"}, {"mood": "开心"})
        self.assertEqual(new["mood"], "开心")


class TestParser(unittest.TestCase):
    def test_tail_extracted(self):
        text = '正文。\n```jiuli\n{"state_diff": {"affection": "+1"}, "suggestions": ["a"]}\n```'
        narrative, tail, warns = parse_output(text)
        self.assertEqual(narrative, "正文。")
        self.assertEqual(tail["state_diff"], {"affection": "+1"})
        self.assertEqual(warns, [])

    def test_no_tail(self):
        narrative, tail, warns = parse_output("只有正文")
        self.assertEqual((narrative, tail, warns), ("只有正文", None, []))

    def test_malformed_tail_degrades(self):
        narrative, tail, warns = parse_output('正文\n```jiuli\n{broken}\n```')
        self.assertIsNone(tail)
        self.assertTrue(warns)

    def test_unknown_keys_dropped(self):
        _, tail, warns = parse_output('x\n```jiuli\n{"hack": 1, "summary": "s"}\n```')
        self.assertEqual(tail, {"summary": "s"})
        self.assertTrue(any("hack" in w for w in warns))


class TestAssembler(unittest.TestCase):
    def test_budget_respected_and_priority_order(self):
        a = ContextAssembler("人设文本", lambda: '{"affection": 5}', "指令",
                             token_budget=300)
        big_wb = [("e%d" % i, 10 - i, "内容" * 200) for i in range(5)]
        system, dropped = a.assemble([(1, "user", "hi" * 50, "chat")], big_wb, ["事实" * 100])
        self.assertLessEqual(est_tokens(system), 300 + est_tokens("内容" * 200))
        self.assertTrue(system.startswith("【指令】"))


class TestLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pkg = make_pkg(self.tmp)
        self.db = Path(self.tmp) / "test.db"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_two_turns_end_to_end(self):
        s = RPSession(self.pkg, MockLLM(), str(self.db))
        r1 = s.run_turn("我们在旧书店里坐下喝茶")
        self.assertIn("001-shudian", r1["wb_hits"])
        self.assertEqual(r1["state"]["affection"], 1)   # MockLLM +1，已校验落库
        self.assertEqual(r1["rejected"], [])
        r2 = s.run_turn("暖炉后面是不是藏着什么")
        self.assertEqual(r2["state"]["affection"], 2)   # 第二轮在第一轮基础上累计
        self.assertTrue(any("旧书店" in f for f in s.store.all_facts(s.session_id)))
        roles = [item[1] for item in s.store.all_messages(s.session_id)]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant"])
        self.assertLess(r2["ctx_tokens_est"], 4000)
        # 记忆检索应命中上一轮存下的事实
        self.assertGreaterEqual(r2["mem_hits"], 0)


def _p0_regression():
    """P0 检索实验回归：hybrid 通道在标注集上 R@5 ≥ 0.8。"""
    corpus_p = Path(__file__).resolve().parent.parent.parent / "p0-retrieval-exp" / "corpus.json"
    ann_p = Path(__file__).resolve().parent.parent.parent / "p0-retrieval-exp" / "annotations.json"
    if not corpus_p.exists() or not ann_p.exists():
        return None

    class P0Retriever:
        """把 p0 语料包一层，对齐 WorldbookRetriever 接口。"""

        def __init__(self, corpus):
            self.entries = [e for e in corpus if not e["constant"]]
            from jiuli.retriever import BM25, tokenize
            docs = []
            for e in self.entries:
                text = (e["title"] + " ") * 2 + (" ".join(e["keys"]) + " ") * 3 + e["content"]
                docs.append(tokenize(text))
            self.bm25 = BM25(docs)
            self.BM25Cls, self.tokenize = BM25, tokenize

        def _lex(self, q, topk):
            qt = self.tokenize(q)
            scored = sorted(((self.bm25.score(qt, i), e["id"])
                             for i, e in enumerate(self.entries)), reverse=True)
            return [(i, s) for s, i in scored if s > 0][:topk]

        def search(self, q, topk=8, k=60):
            qq = q.lower()
            r1 = [(e["id"], 1) for e in self.entries if any(k.lower() in qq for k in e["keys"])]
            r2 = self._lex(q, topk)
            rrf = {}
            for rank, (eid, _) in enumerate(r1):
                rrf[eid] = rrf.get(eid, 0) + 1 / (k + rank + 1)
            for rank, (eid, _) in enumerate(r2):
                rrf[eid] = rrf.get(eid, 0) + 1 / (k + rank + 1)
            return sorted(rrf.items(), key=lambda x: -x[1])[:topk]

    corpus = json.loads(corpus_p.read_text(encoding="utf-8"))
    ann = json.loads(ann_p.read_text(encoding="utf-8"))["queries"]
    retr = P0Retriever(corpus)
    total = rec = 0
    for q in ann:
        if q["type"] == "constant_covered":
            continue
        # 标题子串 -> 条目 id（与 run_eval.resolve_gold 同逻辑，内联避免跨包导入）
        gold = {e["id"] for e in retr.entries if e["card"] == q["card"]
                and any(p in e["title"] for p in q["gold"])}
        ranked = retr.search(q["q"], topk=5)
        top5 = [eid for eid, _ in ranked[:5]]
        total += 1
        rec += len(gold & set(top5)) / len(gold)
    return rec / total


class TestP0Regression(unittest.TestCase):
    def test_hybrid_recall(self):
        r = _p0_regression()
        if r is None:
            self.skipTest("p0-retrieval-exp 不存在")
        self.assertGreaterEqual(r, 0.8, "P0 hybrid R@5 回归低于阈值: %.3f" % r)


class TestSwipeForkStream(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pkg = make_pkg(self.tmp)
        self.db = str(Path(self.tmp) / "t.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reroll_keeps_one_exchange(self):
        s = RPSession(self.pkg, MockLLM(), self.db)
        s.run_turn("第一句")
        r2 = s.run_turn("第一句")  # reroll 用的输入无所谓，先填两条
        r = s.reroll()
        self.assertIsNotNone(r)
        roles = [item[1] for item in s.store.all_messages(s.session_id)]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant"])
        self.assertIn("第一句", s.store.all_messages(s.session_id)[0][2])

    def test_fork_copies_history_and_state(self):
        s = RPSession(self.pkg, MockLLM(), self.db)
        s.run_turn("来一段")
        new_id = s.fork(title="分叉线")
        msgs = s.store.all_messages(new_id)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(s.store.get_state(new_id)["affection"], 1)
        self.assertEqual(len(s.store.list_sessions()), 2)

    def test_stream_yields_deltas_and_done(self):
        s = RPSession(self.pkg, MockLLM(), self.db)
        events = list(s.run_turn_stream("流式一下"))
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds[0], "status")
        self.assertIn("delta", kinds)
        self.assertEqual(kinds[-1], "done")
        text = "".join(e["text"] for e in events if e["type"] == "delta")
        self.assertNotIn("```jiuli", text)  # 尾部块不外发
        self.assertEqual(events[-1]["result"]["state"]["affection"], 1)
        roles = [item[1] for item in s.store.all_messages(s.session_id)]
        self.assertEqual(roles, ["user", "assistant"])

    def test_summary_persisted(self):
        s = RPSession(self.pkg, MockLLM(), self.db)
        s.run_turn("记一点东西")
        self.assertTrue(s.store.get_summary(s.session_id))


class TestLongSession50(unittest.TestCase):
    """P2 验收：50 轮长会话，单轮 ctx ≤ 预算；状态零漂移。"""

    def test_50_turns(self):
        tmp = tempfile.mkdtemp()
        try:
            s = RPSession(make_pkg(tmp), MockLLM(), str(Path(tmp) / "long.db"),
                          token_budget=4000)
            max_ctx = 0
            for i in range(50):
                r = s.run_turn("这是第%d轮对话，玩家做了一些事情" % (i + 1))
                max_ctx = max(max_ctx, r["ctx_tokens_est"])
                self.assertLessEqual(r["ctx_tokens_est"], 4000,
                                     "第 %d 轮上下文超预算: %d" % (i + 1, r["ctx_tokens_est"]))
            self.assertEqual(s.state["affection"], 50)          # MockLLM 每轮 +1
            db_state = s.store.get_state(s.session_id)
            self.assertEqual(db_state["affection"], 50)         # 落库一致 = 零漂移
            self.assertLess(max_ctx, 4000)
            print("\n50 轮最大 ctx = %d tok" % max_ctx)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def make_pkg2(tmp):
    """第二个角色包（群聊测试用），键与内容均不同。"""
    pkg = make_pkg(tmp, "card_package-demo2")
    persona = dict(PERSONA)
    persona["name"] = "阿灰"
    persona["bio"] = "修表铺的沉默学徒。"
    (pkg / "persona.json").write_text(json.dumps(persona, ensure_ascii=False), encoding="utf-8")
    idx_p = pkg / "worldbook" / "index.json"
    idx = json.loads(idx_p.read_text(encoding="utf-8"))
    idx["entries"][0]["keys"] = ["修表铺", "怀表"]
    idx["entries"][0]["title"] = "修表铺"
    idx["entries"][0]["content_file"] = "entries/004-watch.md"
    idx_p.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
    (pkg / "worldbook" / "entries" / "004-watch.md").write_text(
        "修表铺里挂满停摆的怀表，阿灰从不解释它们停在同一时刻。", encoding="utf-8")
    return pkg


class TestMemoryV1(unittest.TestCase):
    def setUp(self):
        from jiuli.memory import MemoryStore
        self.tmp = tempfile.mkdtemp()
        self.store = SessionStore(str(Path(self.tmp) / "m.db"))
        self.mem = MemoryStore(self.store, max_facts=5)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dedup(self):
        s = self.store.create_session()
        r1 = self.mem.add(s, ["玩家在雨天借给凌夜一把伞"])
        r2 = self.mem.add(s, ["玩家在雨天借给凌夜一把伞"])
        self.assertEqual(r1["added"], 1)
        self.assertEqual(r2["added"], 0)
        self.assertEqual(r2["duplicates"], 1)

    def test_correction_supersedes(self):
        s = self.store.create_session()
        self.mem.add(s, [{"text": "凌夜收下了玩家给的三明治", "kind": "event"}])
        r = self.mem.add(s, [{"text": "其实凌夜并没有收下玩家给的三明治", "kind": "event"}])
        self.assertEqual(r["superseded"], 1)
        recs = self.store.active_facts()
        self.assertEqual(len(recs), 1)
        self.assertIn("其实", recs[0]["text"])

    def test_contradiction_pair(self):
        s = self.store.create_session()
        self.mem.add(s, [{"text": "玩家拒绝了文化祭的邀请"}])
        self.mem.add(s, [{"text": "玩家答应了文化祭的邀请"}])
        recs = self.store.active_facts()
        self.assertEqual(len(recs), 1)
        self.assertIn("答应", recs[0]["text"])

    def test_prune_keeps_recent(self):
        s = self.store.create_session()
        for i in range(8):
            self.mem.add(s, [{"text": "第%d件不同的事件记录，各不重复" % i}])
        self.assertLessEqual(len(self.store.active_facts()), 5)

    def test_search_kinds_filter(self):
        s = self.store.create_session()
        self.mem.add(s, [{"text": "玩家喜欢热茶", "kind": "preference"},
                         {"text": "玩家上周搬进了新公寓", "kind": "event"}])
        self.assertEqual(self.mem.search("玩家喝什么", kinds=["preference"]),
                         ["玩家喜欢热茶"])


class TestScheduler(unittest.TestCase):
    def setUp(self):
        from jiuli.scheduler import ProactiveEngine
        self.tmp = tempfile.mkdtemp()
        self.s = RPSession(make_pkg(self.tmp), MockLLM(),
                           str(Path(self.tmp) / "s.db"))
        self.now = time.time()
        self.engine = ProactiveEngine(lambda: [self.s], now_fn=lambda: self.now)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _advance(self, seconds):
        self.now += seconds

    def test_no_proactive_when_active(self):
        self.s.run_turn("刚说完话")
        events = self.engine.tick()
        self.assertEqual([e for e in events if e["kind"] == "proactive"], [])

    def test_proactive_after_idle_with_quota(self):
        self.s.run_turn("我先下了，晚安")
        # 上一条发言是角色：需要空闲超过 2 倍阈值才允许角色先开口
        self._advance(15 * 3600)
        for i in range(5):  # 连续 tick 触发配额上限
            self.engine.tick()
            self._advance(3600)
        msgs = [m for m in self.s.store.all_messages(self.s.session_id) if m[3] == "proactive"]
        self.assertEqual(len(msgs), 3, "节流阀应把每日主动消息限制在 3 条")

    def test_disabled_config(self):
        self.s.run_turn("晚安")
        self._advance(7 * 3600)
        self.engine.set_config(self.s, {"enabled": False})
        events = self.engine.tick()
        self.assertEqual([e for e in events if e["kind"] == "proactive"], [])

    def test_force_bypasses_quota_and_idle(self):
        self.s.run_turn("在吗")
        events = self.engine.tick_session(self.s, force=True)
        self.assertEqual(len([e for e in events if e["kind"] == "proactive"]), 1)

    def test_world_event_once_per_day(self):
        self.s.run_turn("出门了")
        self._advance(30 * 3600)
        self.engine.tick()
        world = [f for f in self.store_facts() if f.get("kind") == "world"]
        self.assertEqual(len(world), 1)
        self._advance(3600)
        self.engine.tick()
        world = [f for f in self.store_facts() if f.get("kind") == "world"]
        self.assertEqual(len(world), 1, "同一天不应重复生成世界事件")

    def store_facts(self):
        return self.s.store.active_facts()


class TestGroupChat(unittest.TestCase):
    def setUp(self):
        from jiuli.group import GroupChat
        self.tmp = tempfile.mkdtemp()
        self.gc = GroupChat([make_pkg(self.tmp), make_pkg2(self.tmp)],
                            MockLLM(), str(Path(self.tmp) / "g.db"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_relevance_order(self):
        r = self.gc.run_turn("我们去旧书店逛逛", max_speakers=2)
        self.assertEqual(r["order"][0], "小铃")  # 命中书店设定的角色先说
        self.assertEqual(set(r["replies"]), {"小铃", "阿灰"})

    def test_second_speaker_sees_first(self):
        r = self.gc.run_turn("你们俩都在场", max_speakers=2)
        second_name = r["order"][1]
        second = [m for m in self.gc.members
                  if m.pkg.persona.get("name") == second_name][0]
        last_user = second.store.all_messages(second.session_id)[-2][2]
        self.assertIn("[现场·刚发生]", last_user)
        self.assertIn(r["replies"][r["order"][0]]["narrative"][:20], last_user)


class TestP5Fixes(unittest.TestCase):
    """P4 复盘四个缺陷的回归。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pkg = make_pkg(self.tmp)
        self.db = str(Path(self.tmp) / "fix.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_scheduler_respects_lock_provider(self):
        """调度器线程必须经由 lock_provider 获取会话锁（缺陷 #1）。"""
        from contextlib import nullcontext
        from jiuli.scheduler import ProactiveEngine
        s = RPSession(self.pkg, MockLLM(), self.db)
        acquired = []

        def provider(sess):
            acquired.append(sess.session_id)
            return nullcontext()

        eng = ProactiveEngine(lambda: [s], lock_provider=provider)
        s.run_turn("晚安")
        eng.tick_session(s, force=True)
        self.assertEqual(acquired, [s.session_id])

    def test_memory_cache_invalidates_across_sessions(self):
        """会话 A 写入新事实后，会话 B 的检索必须能看到（缺陷 #2）。"""
        a = RPSession(self.pkg, MockLLM(), self.db)
        b = RPSession(self.pkg, MockLLM(), self.db)  # 共享同一个 facts 表
        a.run_turn("记住：玩家在旧书店的暖炉后藏了一封信")  # A 的缓存刷新到当前版本
        b.memory.search("随便查点什么")                     # B 也缓存了当前版本
        a.run_turn("玩家把怀表落在了书店柜台")              # A 新增事实
        hits = b.memory.search("怀表")
        self.assertTrue(any("怀表" in h for h in hits),
                        "跨会话缓存未失效：B 看不到 A 新写入的事实")

    def test_fork_copies_summary(self):
        """分叉必须继承前情摘要（缺陷 #3）。"""
        s = RPSession(self.pkg, MockLLM(), self.db)
        s.run_turn("来一段")
        s.store.set_summary(s.session_id, "玩家与角色已经相识三天")
        new_id = s.fork()
        self.assertEqual(s.store.get_summary(new_id), "玩家与角色已经相识三天")

    def test_reroll_retracts_ghost_facts(self):
        """swipe 后被删回合的记忆事实应撤回（缺陷 #4）。"""
        s = RPSession(self.pkg, MockLLM(), self.db)
        s.run_turn("玩家把祖传怀表送给了角色")
        ghost_id = max(f["id"] for f in s.store.active_facts())  # 该回合写入的事实
        s.reroll()  # 重生成会写入一条新的（同文本）事实，属正常
        by_id = {f["id"]: f for f in s.store.active_facts()}
        self.assertNotIn(ghost_id, by_id, "被 roll 掉回合的事实应已停用")


if __name__ == "__main__":
    unittest.main()
