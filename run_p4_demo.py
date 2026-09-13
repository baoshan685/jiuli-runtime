# -*- coding: utf-8 -*-
"""P4 真模型演示：记忆纠错消解 / 30 天回归北极星 / 双角色群聊。"""
import json
import sys
import tempfile
from pathlib import Path

WS = Path(__file__).resolve().parent
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS.parent / "jiuli-runtime" / "tests") if False else str(WS))

from jiuli.llm import OpenAICompatClient  # noqa: E402
from jiuli.loop import RPSession  # noqa: E402
from jiuli.scheduler import ProactiveEngine  # noqa: E402
from jiuli.group import GroupChat  # noqa: E402

BASE_URL = __import__("os").environ.get("JIULI_BASE_URL", "https://api.teamorouter.cn/v1")
API_KEY = __import__("os").environ.get("JIULI_API_KEY", "")
MODEL = __import__("os").environ.get("JIULI_MODEL", "glm-5.3-flash-free")
PKG_LY = WS.parent / "jiuli-real-test" / "pkg"
DEMO_DB = WS / "p4_demo.db"


def build_pkg_xiaoling(tmp):
    """第二个真实人设包：旧书店图书管理员小铃（凌夜世界观外的独立角色）。"""
    pkg = tmp / "card_package-xiaoling"
    ent = pkg / "worldbook" / "entries"
    ent.mkdir(parents=True)
    persona = {"name": "小铃", "short_name": "小铃",
               "bio": "巷子深处旧书店的管理员，爱笑，喜欢旧书和热茶。",
               "description": "", "personality": "温柔好奇", "scenario": "旧书店",
               "tags": []}
    (pkg / "persona.json").write_text(json.dumps(persona, ensure_ascii=False), encoding="utf-8")
    (pkg / "manifest.json").write_text(json.dumps({"schema_version": "1.0.0"}), encoding="utf-8")
    (pkg / "state_machine.json").write_text(json.dumps({"variables": [
        {"name": "affection", "label": "好感度", "default": 0, "min": 0, "max": 100,
         "explicit": True}]}), encoding="utf-8")
    entries = [
        {"id": "001-shudian", "title": "书店设定", "keys": ["旧书店", "书店", "拾光"],
         "constant": False, "enabled": True, "content_file": "entries/001-shudian.md",
         "content": "书店名叫「拾光」，只收旧书。冬天烧一只旧暖炉，木地板会响。小铃会给常客留 position——靠窗第三排书架后那把藤椅。"},
    ]
    (pkg / "worldbook" / "index.json").write_text(json.dumps({"entries": entries},
        ensure_ascii=False), encoding="utf-8")
    for e in entries:
        (ent / (e["id"] + ".md")).write_text(e["content"], encoding="utf-8")
    return pkg


def show_facts(store, title):
    print("\n--- 记忆库现状（%s）---" % title)
    for r in store.active_facts():
        print("  [%s] %s" % (r["kind"], r["text"][:60]))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="p4_demo_"))
    llm = OpenAICompatClient(BASE_URL, API_KEY, MODEL, timeout=240)

    # ═══ 1) 记忆纠错消解 ═══
    print("=" * 30, "\n[1] 记忆纠错消解（玩家 retcon）")
    db = DEMO_DB
    if db.exists():
        db.unlink()
    s = RPSession(PKG_LY, llm, str(db), token_budget=4000)
    s.run_turn("我把一份文化祭的策划案交给了凌夜，请她帮忙排版")
    before = len(s.store.active_facts())
    s.run_turn("等等，其实策划案我后来没有交给凌夜，我给错了人，交给了天宫慧")
    after = s.store.active_facts()
    print("  纠错前生效事实: %d 条 → 纠错后: %d 条（旧事实应停用、新事实继承）" % (before, len(after)))

    # ═══ 2) 北极星：30 天后回归 ═══
    print("\n" + "=" * 30, "\n[2] 北极星演示：放下 30 天后回来")
    # 模拟时间流逝：把既有事实的时间戳改到 30 天前（仅演示；真实使用中由自然时间积累）
    import sqlite3
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE facts SET created = datetime('now','-30 days')")
    conn.commit()
    conn.close()
    s_b = RPSession(PKG_LY, llm, str(db), token_budget=4000)  # 全新会话
    r = s_b.run_turn("（一个月后推门进来）好久没来了。最近怎么样？")
    print("  玩家: （一个月后推门进来）好久没来了。最近怎么样？")
    print("  凌夜>", r["narrative"][:220].replace("\n", " "))
    print("  记忆命中:", r["mem_hits"], " 世界书命中:", r["wb_hits"])

    engine = ProactiveEngine(lambda: [s_b])
    events = engine.tick_session(s_b, force=True)
    for e in events:
        if e["kind"] == "proactive":
            print("  [主动消息] 凌夜>", e["message"][:160].replace("\n", " "))

    # ═══ 3) 双角色群聊 ═══
    print("\n" + "=" * 30, "\n[3] 多 agent 群聊（凌夜 × 小铃）")
    pkg_xl = build_pkg_xiaoling(tmp)
    gc = GroupChat([PKG_LY, pkg_xl], llm, str(db), token_budget=4000)
    gr = gc.run_turn("我同时认识你们两个。今天想找你们聊聊各自最近的生活", max_speakers=2)
    for name in gr["order"]:
        rep = gr["replies"][name]
        print("  %s>" % name, rep["narrative"][:150].replace("\n", " "))

    print("\nP4 演示完成 ✔")


if __name__ == "__main__":
    main()
