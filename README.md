# 酒醴（jiuli）v0.1.0 —— 瘦上下文本地角色扮演引擎

**把酒馆角色卡蒸馏成结构化语义包，用确定性的运行时驱动"有记忆、有生活、成本可控"的角色扮演。**

> 比 RisuAI 多生命感，比酒馆少折腾，比托管平台自由。
>
> 设计依据：[docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) ·
> 检索实验：[docs/P0_RETRIEVAL_REPORT.md](docs/P0_RETRIEVAL_REPORT.md) ·
> 验收数据：[P2_ACCEPTANCE.md](P2_ACCEPTANCE.md) ·
> 已知问题：[RELEASE.md](RELEASE.md)

## 亮点（全部有实测数据）

- **输入成本约为酒馆重度配置的 1/5～1/10**：每轮输入 508→3078 token（8 轮实测），世界书按需命中注入，而非全量塞入；
- **状态零漂移**：50 轮长会话断言通过——模型只提 diff，合法性由代码裁决，未知变量 / 越界 / 超长一律拒收；
- **跨会话记忆 + 纠错消解**：玩家 retcon 后旧事实自动停用（词面规则 + LLM NLI 裁判兜底）；放下 30 天回来，角色记得一切并主动开口（北极星实测通过）；
- **中文改写检索**：玩家说"那个马尾辫女生"也能命中世界书——字面触发 + 字符 bigram BM25 混合，把酒馆式字面触发 8.7% 的改写召回拉到 82.6%；
- **主动性调度器**：空闲阈值 + 每日配额节流阀 + 离线世界事件，角色会自己发消息；
- **多 agent 群聊 v1**：每角色独立上下文 / 状态 / 记忆，检索相关性裁判排序发言；
- **零第三方依赖**：Python ≥ 3.8 标准库实现全部功能（含 Web 服务器与 SSE 流式）。

## 快速开始

```bash
# ① 离线演示（内置示例卡，无需 API key，秒开）
python -m jiuli.server --pkg examples/demo-pkg --mock
# 打开 http://127.0.0.1:8770

# ② 接真实模型（任意 OpenAI 兼容端点；也可在 UI「模型设置」里热配置）
python -m jiuli.server --pkg examples/demo-pkg \
    --api-base https://api.xxx.com/v1 --api-key $JIULI_API_KEY --model your-model

# ③ 纯 CLI（无浏览器场景）
python -m jiuli.cli --pkg examples/demo-pkg --mock

# ④ 一键启动（默认即内置示例卡）
#   Windows: start.bat        macOS/Linux: ./start.sh        Docker: docker compose up -d

# ⑤ 测试（46 项：检索回归 / 50 轮预算验收 / HTTP 冒烟 / 记忆与调度）
python -m unittest discover tests
```

## Web UI 功能

- **流式聊天**：SSE 逐字输出，结构化尾部块不外发到正文；
- **状态面板**：每轮实时显示状态变量变化与被拒 diff；
- **存档墙**：多会话管理 + 一键分叉（复制历史 / 状态 / 摘要到新线）；
- **swipe 重roll**：重新生成最后一轮，并自动撤回该轮写入的记忆事实；
- **插图画廊**：自动读取语义包 illustrations.json + 本地图库；
- **模型向导**：热配置 API / 一键探测本地 Ollama / 连通性测试；
- **主动性**：角色可主动发来消息（带徽标），也可手动触发。

## 架构（一个回合发生什么）

```
玩家输入
  → 检索：世界书 hybrid 命中 + 记忆命中（无命中兜底最近事实）(retriever/memory)
  → 组装：指令+人设+状态+摘要+命中+近12轮，预算内裁剪      (context)
  → 生成：单次 LLM 调用（可流式），叙事 + ```jiuli 尾部块    (llm)
  → 解析：state_diff / illustration / memory_facts / summary (parser)
  → 校验：未知变量拒收、越界拒收、字符串超长拒收             (state)
  → 落库：消息/状态/记忆/摘要 全部进 SQLite                  (session)
```

关键原则：**数值状态与存档是运行时的职责，不是模型的自觉。**
模型输出的 diff 只是提议；合法性由 state.py 裁决，解析失败降级为纯叙事并告警。

## 卡片语义包（schema v1）

由蒸馏器 `generate_package.py` 产出，运行时只消费这个目录：

```
manifest.json / persona.json / state_machine.json / routes.json
preset_intents.json / regex_semantics.json / illustrations.json
worldbook/index.json + worldbook/entries/*.md
```

`state_machine.json` 中 `explicit=False` 的变量（关键词低置信提示）运行时**默认不启用**，
需蒸馏确认或配置放行——这是检索实验发现的误报问题的第一道闸。

### 把你的酒馆卡搬进来

用姊妹项目 [tavern-card-distiller](https://github.com/leigegehaha/tavern-card-distiller)
（酒馆卡蒸馏器，V1/V2/V3 全格式无损解析）：

```bash
python extract_card.py <card.png> -o out/
python generate_package.py out/card_data.json
```

## 开发者验收脚本

`run_p2_acceptance.py` / `run_p4_demo.py` / `run_migration_regression.py` 是真实模型
验收脚本（需环境变量 `JIULI_API_KEY`，以及本机存在的蒸馏器仓库与评测标注），
普通使用不需要它们；11 卡迁移回归的最近结果见 `migration_regression_report.json`
（11/11 卡打包与回合通过，检索标注 5 卡 1.0）。

## 隐私与依赖

- 对话、状态、记忆全部存本地 SQLite；**无遥测**；API key 仅存在于进程内存；
- 零第三方依赖；Windows / macOS / Linux / Docker。
