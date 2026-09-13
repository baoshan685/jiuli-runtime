# 酒醴（jiuli）运行时 v0.1 —— 开发者预览

瘦上下文本地 RP 引擎。P2 验收数据见 `P2_ACCEPTANCE.md`（50 轮 ctx≤4K、状态零漂移、
跨会话记忆、输入成本约为酒馆重度配置 1/5～1/10）；设计依据见 `../jiuli_development_plan.md`，
检索算法依据见 `../p0-retrieval-exp/REPORT.md`。

## 定位（v0.1 已实现 / 未实现）

| 已实现 | 未实现（按计划属于后续阶段） |
|---|---|
| 世界书双通道检索（key + BM25 hybrid） | embedding 第三通道（需模型下载/API） |
| 瘦上下文组装器（预算裁剪 + 会话摘要） | 摘要压缩质量压测 |
| 状态机校验（diff 提议 / 代码裁决 / max_len） | 主动性调度器 / 多 agent 群聊 |
| SQLite 存档 + swipe 重roll + 分叉 | Web UI 美化 / 分支树可视化 |
| 跨会话记忆库 + 检索注入 + 兜底注入 | 记忆来源标记与冲突消解 |
| **Web UI**（流式聊天/状态面板/存档墙/插图/模型向导） | 11 卡全量迁移回归 |
| MockLLM 离线演示 + OpenAI 兼容接入 | 模型接入向导持久化（当前仅内存） |

### P4 已实现（记忆 v1 / 主动性 / 群聊）

| 能力 | 说明 |
|---|---|
| 记忆去重 | bigram Jaccard ≥ 0.7 不重复入库 |
| 纠错消解 | 纠正线索（其实/不是/误会…，含玩家原话）+ 重叠系数 → 旧事实停用（retcon）；词面不中时 LLM NLI 裁判兜底 |
| 遗忘修剪 | 生效事实超过 500 条从最旧停用 |
| 来源标记 | facts 带 kind（event/relationship/world/preference），检索可按类过滤 |
| 主动消息 | 空闲阈值 + 每日配额节流阀 + 上一发言者规则，UI 轮询展示（带"主动发来"徽标） |
| 离线演化 | 空闲超 1 天生成世界事件入记忆（每自然日至多一次） |
| 多 agent 群聊 | 每角色独立上下文/状态，检索相关性排序发言（v1 限制见 RELEASE.md） |
| 北极星 | 实测通过：放下 30 天回来，角色引用纠错后的记忆自然接续剧情并主动开口 |

## 快速开始（零第三方依赖，Python ≥ 3.8）

```bash
# ① 离线演示（无 API key，秒开）
python -m jiuli.server --pkg <card_package目录> --mock
# 打开 http://127.0.0.1:8770

# ② 接真实模型（任意 OpenAI 兼容端点；也可在 UI 的「模型设置」里热配置）
python -m jiuli.server --pkg <pkg> --api-base https://api.xxx.com/v1 \
    --api-key sk-xxx --model your-model

# ③ 纯 CLI（无浏览器场景）
python -m jiuli.cli --pkg <pkg> --mock

# ④ 一键启动
#   Windows: start.bat        macOS/Linux: ./start.sh
#   Docker:  docker compose up -d
```

## Web UI 功能

- **流式聊天**：SSE 逐字输出，尾部块（状态/记忆/建议）不外发到正文；
- **状态面板**：每轮实时显示状态变量与被拒 diff；
- **存档墙**：多会话管理 + 一键分叉（复制历史与状态到新线）；
- **swipe 重roll**：删除最后一组交换重新生成；
- **插图画廊**：自动读取语义包 illustrations.json + 本地图库；
- **模型向导**：热配置 API / 一键探测本地 Ollama / 连通性测试。

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

由 `generate_package.py` 产出，运行时只消费这个目录：

```
manifest.json / persona.json / state_machine.json / routes.json
preset_intents.json / regex_semantics.json / illustrations.json
worldbook/index.json + worldbook/entries/*.md
```

`state_machine.json` 中 `explicit=False` 的变量（关键词低置信提示）运行时**默认不启用**，
需蒸馏确认或配置放行——这是 P0 实验发现的误报问题的第一道闸。
