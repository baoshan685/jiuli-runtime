# -*- coding: utf-8 -*-
"""11 卡全量迁移回归：仓库每张真实卡 -> 语义包 -> 运行时加载 -> MockLLM 回合。

对有标注的卡额外计算 hybrid 检索 recall@5（标注来自 p0-retrieval-exp）。
产出：stdout 表格 + migration_regression_report.json
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

WS = Path(__file__).resolve().parent
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS.parent / "p0-retrieval-exp"))
sys.path.insert(0, str(WS.parent / "tavern-card-distiller" / "scripts"))

from build_corpus import parse_worldbook  # noqa: E402
from jiuli.llm import MockLLM  # noqa: E402
from jiuli.loop import RPSession  # noqa: E402
from jiuli.retriever import tokenize  # noqa: E402

REPO = WS.parent / "tavern-card-distiller"
ANNOT = WS.parent / "p0-retrieval-exp" / "annotations.json"


def build_package(card_dir, workdir):
    """真实世界书 -> card_data.json -> generate_package.py。返回 pkg 目录。"""
    entries = parse_worldbook(card_dir / "references" / "world_book.md")
    skill_md = (card_dir / "SKILL.md").read_text(encoding="utf-8")
    first_mes = ""
    if "## 默认开场白" in skill_md:
        first_mes = skill_md.split("## 默认开场白", 1)[1].split("## ", 1)[0].strip()
    cd = workdir / "card_data.json"
    cd.write_text(json.dumps({
        "spec": "v2", "spec_version": "chara_card_v2",
        "name": card_dir.name[3:],
        "description": "（角色设定存放在世界书中）", "personality": "",
        "first_mes": first_mes, "mes_example": "", "scenario": "",
        "system_prompt": "", "post_history_instructions": "",
        "creator_notes": "", "creator": "", "character_version": "",
        "tags": [], "alternate_greetings": [],
        "world_book": [{"keys": e["keys"], "content": e["content"],
                        "enabled": True, "constant": e["constant"],
                        "name": e["title"], "comment": e["title"]} for e in entries],
        "regex_scripts": [], "embedded_images": [], "assets": [], "extensions": {},
        "embedded_image_count": 0,
    }, ensure_ascii=False), encoding="utf-8")
    pkg = workdir / ("pkg-" + card_dir.name)
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "generate_package.py"),
         str(cd), "--output-dir", str(pkg), "--skill-name", card_dir.name],
        capture_output=True, text=True)
    return pkg, r.returncode == 0, len(entries)


def recall_for_card(pkg, queries):
    """该卡标注查询的 hybrid recall@5（标题子串解析 gold）。"""
    from jiuli.retriever import WorldbookRetriever
    retr = WorldbookRetriever(pkg)
    total = rec = 0
    for q in queries:
        if q["type"] == "constant_covered":
            continue
        gold = {e["id"] for e in retr.entries if any(p in e["title"] for p in q["gold"])}
        if not gold:
            continue
        ranked = retr.search(q["q"], topk=5)
        total += 1
        rec += len(gold & {eid for eid, _ in ranked[:5]}) / len(gold)
    return (rec / total, total) if total else (None, 0)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="mig_regress_"))
    ann = json.loads(ANNOT.read_text(encoding="utf-8"))["queries"] if ANNOT.exists() else []
    results, all_pass = [], True

    for card_dir in sorted(REPO.glob("rp-*")):
        if not card_dir.is_dir():
            continue
        row = {"card": card_dir.name}
        try:
            pkg, ok, n_entries = build_package(card_dir, tmp)
            row["package_built"] = ok
            row["worldbook_entries"] = n_entries
            if not ok:
                all_pass = False
                results.append(row)
                continue

            retr_rows = json.loads((pkg / "worldbook" / "index.json").read_text(encoding="utf-8"))["entries"]
            row["nonconstant_entries"] = sum(1 for e in retr_rows if not e["constant"])

            # MockLLM 回合必须无异常
            s = RPSession(pkg, MockLLM(), str(tmp / (card_dir.name + ".db")), token_budget=4000)
            r = s.run_turn("我们随便聊两句，讲讲这里的事")
            row["mock_turn_ok"] = bool(r["narrative"])
            row["ctx_est"] = r["ctx_tokens_est"]

            qs = [q for q in ann if q["card"] == card_dir.name]
            if qs:
                rec, n = recall_for_card(pkg, qs)
                row["annotated_recall5"] = None if rec is None else round(rec, 3)
                row["annotated_n"] = n
                if rec is not None and rec < 0.8:
                    all_pass = False
            results.append(row)
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)[:120]
            all_pass = False
            results.append(row)

    print("%-16s %-6s %-6s %-8s %-10s %-8s" %
          ("卡", "打包", "条目", "非常驻", "mock回合", "recall@5"))
    for r in results:
        print("%-16s %-6s %-6s %-8s %-10s %-8s" % (
            r["card"], "✓" if r.get("package_built") else "✗",
            r.get("worldbook_entries", "-"), r.get("nonconstant_entries", "-"),
            "✓" if r.get("mock_turn_ok") else ("✗" if "error" in r else "-"),
            r.get("annotated_recall5", "-")))
        if "error" in r:
            print("    error:", r["error"])

    (WS / "migration_regression_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n总体: %s （明细 -> migration_regression_report.json）"
          % "PASS ✔" if all_pass else "\n总体: FAIL ✗")


if __name__ == "__main__":
    main()
