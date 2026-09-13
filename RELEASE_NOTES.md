# 酒醴（jiuli）v0.1.0 —— 开发者预览

**瘦上下文本地角色扮演引擎**：把酒馆角色卡蒸馏成结构化语义包，用确定性的运行时驱动"有记忆、有生活、成本可控"的角色扮演。

定位语：比 RisuAI 多生命感，比酒馆少折腾，比托管平台自由。

## 亮点（全部有实测数据背书）

- **输入成本约为酒馆重度配置的 1/5～1/10**：每轮输入 508→3078 token（8 轮实测），世界书按需命中注入
- **状态零漂移**：50 轮长会话断言通过；模型只提 diff，合法性由代码裁决（未知变量/越界/超长一律拒收）
- **跨会话记忆 + 纠错消解**：玩家 retcon 后旧事实自动停用（词面规则 + LLM NLI 裁判）；放下 30 天回来角色记得一切并主动开口（北极星实测通过）
- **主动性调度器**：空闲阈值 + 每日配额节流阀 + 离线世界事件
- **多 agent 群聊 v1**：每角色独立上下文，检索相关性裁判排序
- **Web UI**：流式聊天 / 状态面板 / 存档墙 / swipe / 分叉 / 插图画廊 / 模型向导（零第三方依赖，stdlib 实现）
- **世界书双通道检索**：酒馆式字面触发 + 字符 bigram BM25 混合（中文改写查询召回 8.7% → 82.6%，详见 docs/P0_RETRIEVAL_REPORT.md）

## 三分钟上手

```bash
python -m jiuli.server --pkg examples/demo-pkg --mock   # 内置示例卡，无需 API key
# 打开 http://127.0.0.1:8770
```

接真实模型：`--api-base <url> --api-key <key> --model <model>`（任意 OpenAI 兼容端点），或在 UI「模型设置」里热配置。

## 把你的酒馆卡搬进来

用姊妹项目 [tavern-card-distiller](https://github.com/leigegehaha/tavern-card-distiller)：
`extract_card.py` 无损解析 PNG/JPEG/WEBP/JSON 卡（V1/V2/V3），`generate_package.py` 产出语义包。

## 质量

- 46 项自动化测试全绿（含 HTTP 冒烟、50 轮预算验收、检索回归、11 卡迁移回归）
- 已知问题如实公开于 RELEASE.md（reasoning 模型开销、摘要覆盖率、群聊无共享场景等 5 项）

## 隐私

对话、状态、记忆全部存本地 SQLite；无遥测；API key 仅存在于进程内存。

## 兼容性

Python ≥ 3.8，零第三方依赖。Windows / macOS / Linux / Docker。