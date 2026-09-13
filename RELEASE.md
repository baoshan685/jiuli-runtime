# 公测发布清单（RELEASE v0.1）

> 状态：功能冻结候选。发布前逐项确认。

## 必须完成

- [ ] 仓库拆分：`jiuli-runtime/` 独立成 repo（当前依赖路径：`../p0-retrieval-exp` 回归测试改为可选跳过 ✓ 已实现；`../jiuli-real-test/pkg` 演示包需内置一个离线示例包）
- [ ] 内置示例卡：随仓库附 1 个可直接 `--mock` 演示的语义包（合成卡即可，不含版权内容）
- [ ] LICENSE（MIT）+ 角色卡来源声明
- [ ] 隐私声明：对话/记忆/状态全部存本地 SQLite，无遥测；API key 仅内存（写明）
- [ ] 版本号统一：`jiuli/__init__.py` `__version__` 与 README/发布页一致
- [ ] 测试全绿：`python -m unittest discover tests`（32 项，含 P0 检索回归）
- [ ] README 快速开始三行命令实测可跑（Windows + 一台 mac）

## 建议完成

- [ ] requirements.txt（空表 + 注释"零依赖"也是声明）
- [ ] screenshots：Web UI 三张图（聊天/状态面板/插图画廊）
- [ ] 迁移漏斗文案：tavern-card-distiller README 顶部加「想要会记生命的角色？看酒醴」
- [ ] issue 模板：bug / 卡片兼容性反馈（附 card_package 的 manifest 而非卡本身）

## 已知问题（发布页如实列出）

1. 模型偶发在 memory_facts 虚构"前史"事实（已收紧 prompt，仍有残留）；
2. 摘要压缩依赖模型尾部块自愿提供，长会话后期覆盖率待压测；
3. 群聊 v1 无共享场景：各角色在各自场景中回应，合并叙事由用户自行拼合；
4. reasoning 模型 completion 开销大（900-6000 tok/轮），推荐非推理模型或限制思考预算；
5. 模型配置不持久化（重启需重填）；词面+NLI 之外的记忆矛盾（数值型）未覆盖。

## 发布渠道

1. GitHub repo + Release tag v0.1.0
2. tavern-card-distiller 交叉引流（迁移漏斗）
3. 酒馆社区帖：标题建议「酒馆卡蒸馏 + 会记生命的本地引擎，不部署酒馆也能玩」
4. 演示素材：北极星脚本 `run_p4_demo.py` 的输出截图（30 天回归 + 傲娇主动消息）
