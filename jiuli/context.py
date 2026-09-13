# -*- coding: utf-8 -*-
"""上下文组装器：瘦上下文的心脏。

每轮注入 = 指令 + 人设 + 当前状态 + 记忆检索命中 + 世界书检索命中 + 近 N 轮对话。
按优先级与预算裁剪：预算超限时先砍记忆/世界书的低分命中，再砍最旧的历史轮，
人设与指令永不裁剪。

预算按「估算 token」计算：CJK 混排按 chars*0.6 估计（工程近似，够用且稳定）。
"""

def est_tokens(text):
    return max(1, int(len(text) * 0.6))


class ContextAssembler:
    def __init__(self, persona, state_line_fn, instruction,
                 token_budget=4000, recent_turns_max=12):
        self.persona = persona                      # str，不裁剪
        self.instruction = instruction              # str，不裁剪
        self.state_line_fn = state_line_fn          # () -> str
        self.token_budget = token_budget
        self.recent_turns_max = recent_turns_max

    def assemble(self, recent_messages, wb_hits, memory_hits,
                 summary="", older_count=0):
        """wb_hits: [(id, score, content)]；memory_hits: [str]

        summary: 会话摘要（存在更早历史时注入，截断到 ~600 字符）。
        older_count: 近窗之外仍存在的历史条数（>0 时摘要才有意义）。
        返回 (system_text, dropped_note)。
        """
        fixed = []
        fixed.append("【指令】" + self.instruction)
        fixed.append("【人设】" + self.persona)
        state_line = self.state_line_fn()
        if state_line:
            fixed.append("【当前状态】" + state_line)
        if summary and older_count > 0:
            fixed.append("【前情摘要（更早剧情）】" + summary[:600])
        used = sum(est_tokens(s) for s in fixed)

        # 世界书命中：按分数降序装入
        wb_used = []
        wb_sorted = sorted(wb_hits, key=lambda x: -x[1])
        for eid, score, content in wb_sorted:
            block = "【设定·%s】%s" % (eid, content)
            cost = est_tokens(block)
            if used + cost > self.token_budget * 0.6:
                break
            wb_used.append(block)
            used += cost

        # 记忆命中
        mem_used = []
        for fact in memory_hits:
            block = "【记忆】" + fact
            cost = est_tokens(block)
            if used + cost > self.token_budget * 0.7:
                break
            mem_used.append(block)
            used += cost

        # 近 N 轮对话：从最新往回装入，超预算或超条数即停
        # 消息元组为 (id, role, content, kind)
        history = []
        for item in reversed(recent_messages[-self.recent_turns_max:]):
            role, content = item[1], item[2]
            block = ("%s: %s" % ("玩家" if role == "user" else "角色", content))
            cost = est_tokens(block)
            if used + cost > self.token_budget:
                break
            history.append((role, block))
            used += cost
        history.reverse()

        parts = fixed + wb_used + mem_used + [b for _, b in history]
        dropped = (len(wb_sorted) - len(wb_used),
                   max(0, len(recent_messages) - len(history)))
        return "\n\n".join(parts), dropped
