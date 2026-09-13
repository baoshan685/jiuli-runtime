# -*- coding: utf-8 -*-
"""多 agent 群聊 v1：每个角色独立上下文/状态/记忆，裁判排序发言。

裁判 v0（确定性）：按各角色检索器对玩家输入的 hybrid 命中分排序，
得分最高的 max_speakers 个角色本轮发言；并列时保持注册顺序。
每个发言角色能看到本轮先发言角色的内容（以文本并入其输入，v1 简化）。

已知限制（记入 RELEASE）：群聊没有全局场景裁判 LLM；跨角色记忆共享全局
facts 库；角色间不知道彼此的私有状态。
"""
import re

from jiuli.loop import RPSession
from jiuli.parser import parse_output


class GroupChat:
    def __init__(self, pkg_dirs, llm, db_path, token_budget=4000):
        self.members = [RPSession(p, llm, db_path, token_budget=token_budget)
                        for p in pkg_dirs]

    def names(self):
        return [m.pkg.persona.get("name", "角色%d" % i)
                for i, m in enumerate(self.members)]

    def _relevance(self, member, user_input):
        hits = member.pkg.retriever.search(user_input, topk=3)
        return sum(score for _, score in hits)

    def run_turn(self, user_input, max_speakers=2):
        """返回 {"order": [名字], "replies": {名字: result}}。"""
        ranked = sorted(self.members,
                        key=lambda m: -self._relevance(m, user_input))
        speakers = ranked[:max_speakers]
        order, replies, transcript = [], {}, ""
        for m in speakers:
            name = m.pkg.persona.get("name", "角色")
            seen = (user_input if not transcript else
                    user_input + "\n[现场·刚发生] " + transcript.strip())
            r = m.run_turn(seen)
            replies[name] = r
            order.append(name)
            first_sentence = re.split(r"[。！？\n]", r["narrative"])[0][:120]
            transcript += "%s说：%s " % (name, first_sentence)
        return {"order": order, "replies": replies}

    def proactive_all(self):
        """所有成员的会话都暴露给调度器（群聊里的角色也可以主动发言）。"""
        return list(self.members)
