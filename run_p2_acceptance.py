# -*- coding: utf-8 -*-
"""P2 验收实测：真模型 8 轮长会话（用量采集）+ 跨会话记忆演示。

产出：stdout 明细 + p2_acceptance_results.json
"""
import json
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parent
sys.path.insert(0, str(WS))
from jiuli.llm import OpenAICompatClient  # noqa: E402
from jiuli.loop import RPSession  # noqa: E402

BASE_URL = __import__("os").environ.get("JIULI_BASE_URL", "https://api.teamorouter.cn/v1")
API_KEY = __import__("os").environ.get("JIULI_API_KEY", "")
MODEL = __import__("os").environ.get("JIULI_MODEL", "glm-5.3-flash-free")
PKG = WS.parent / "jiuli-real-test" / "pkg"

TURNS_A = [
    "我放学后绕到办公室，正好听见老师在和凌夜谈话",
    "第二天在旧书店那条街的巷口碰到她",
    "天宫慧又在教室里上演她的漫画桥段了",
    "我帮凌夜把便利店的货搬进了后仓",
    "体育课上她一个人坐在场边，我走过去搭话",
    "放学路上我们聊到了家里的事，她破天荒没有转移话题",
    "文化节要到了，班上在讨论出什么节目",
    "记得我们这几天都经历了什么吗？挑三件重要的说",
]


def turns(llm, tag, session, inputs):
    rows = []
    for i, line in enumerate(inputs, 1):
        t0 = time.time()
        r = session.run_turn(line)
        dt = time.time() - t0
        u = r.get("usage") or {}
        rows.append({"tag": tag, "turn": i, "ctx_est": r["ctx_tokens_est"],
                     "prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens"),
                     "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                     "sec": round(dt, 1), "wb": len(r["wb_hits"]), "mem": r["mem_hits"],
                     "applied": r["applied"], "narr_len": len(r["narrative"])})
        print(f"[{tag}#{i}] ctx={r['ctx_tokens_est']} prompt={u.get('prompt_tokens')} "
              f"completion={u.get('completion_tokens')} {dt:.1f}s wb={len(r['wb_hits'])} "
              f"mem={r['mem_hits']} applied={r['applied'] or '-'}")
        if r["warnings"]:
            print("   warn:", r["warnings"])
    return rows


def main():
    db = WS / "p2_acceptance.db"
    if db.exists():
        db.unlink()
    llm = OpenAICompatClient(BASE_URL, API_KEY, MODEL, timeout=240)
    s = RPSession(PKG, llm, str(db), token_budget=4000)

    print("== 会话 A：8 轮 ==")
    rows = turns(llm, "A", s, TURNS_A)

    print("\n== 会话 B：跨会话记忆 ==")
    s2 = RPSession(PKG, llm, str(db), token_budget=4000)  # 新 session，共享记忆库
    r = s2.run_turn("好久没来书店了。还记得我们上次相处时发生过哪些事吗？说两件你印象最深的")
    rows.append({"tag": "B-cross", "turn": 1, "ctx_est": r["ctx_tokens_est"],
                 "prompt": (r.get("usage") or {}).get("prompt_tokens"),
                 "completion": (r.get("usage") or {}).get("completion_tokens"),
                 "reasoning": ((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
                 "sec": None, "wb": len(r["wb_hits"]), "mem": r["mem_hits"],
                 "applied": r["applied"], "narr_len": len(r["narrative"])})
    print("[B] mem命中:", r["mem_hits"], "叙述片段:", r["narrative"][:160].replace("\n", " "))

    out = {"rows": rows, "facts_total": len(s.store.all_facts()),
           "sessions": s.store.list_sessions()}
    (WS / "p2_acceptance_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n结果已写入 p2_acceptance_results.json")


if __name__ == "__main__":
    main()
