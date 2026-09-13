# -*- coding: utf-8 -*-
"""命令行入口。

用法：
  python -m jiuli.cli --pkg <card_package目录> --mock          # 离线演示
  python -m jiuli.cli --pkg <pkg> --api-base URL --api-key K --model M
交互命令：/state 查看状态 · /quit 退出
"""
import argparse
import sys

from jiuli.llm import MockLLM, OpenAICompatClient
from jiuli.loop import RPSession


def main():
    ap = argparse.ArgumentParser(prog="jiuli")
    ap.add_argument("--pkg", required=True, help="卡片语义包目录")
    ap.add_argument("--db", default="jiuli.db", help="SQLite 路径")
    ap.add_argument("--mock", action="store_true", help="离线演示模式（无 API）")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--budget", type=int, default=4000, help="上下文 token 预算")
    args = ap.parse_args()

    if args.mock:
        llm = MockLLM()
    elif args.api_base and args.model:
        llm = OpenAICompatClient(args.api_base, args.api_key or "", args.model)
    else:
        print("需要 --mock 或 --api-base/--model", file=sys.stderr)
        sys.exit(2)

    session = RPSession(args.pkg, llm, args.db, token_budget=args.budget)
    print("=== 酒醴 v0.1（%s / %s）===" % (
        session.pkg.persona.get("name", "?"), "mock" if args.mock else args.model))
    print("状态变量初始值: %s  ·  /state 查看 · /quit 退出" % session.state)
    while True:
        try:
            line = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/state":
            print("state =", session.state)
            continue
        r = session.run_turn(line)
        print("\n角色> " + r["narrative"])
        if r["applied"]:
            print("[状态变化] %s" % r["applied"])
        if r["rejected"]:
            print("[状态被拒] %s" % r["rejected"])
        if r["illustration"]:
            print("[插图] %s" % r["illustration"])
        if r["suggestions"]:
            print("接下来可以：")
            for i, s in enumerate(r["suggestions"][:3], 1):
                print("  %d. %s" % (i, s))
        print("(ctx≈%s tok, wb=%s, mem=%s%s)" % (
            r["ctx_tokens_est"], r["wb_hits"], r["mem_hits"],
            (", warn=%d" % len(r["warnings"])) if r["warnings"] else ""))


if __name__ == "__main__":
    main()
