# -*- coding: utf-8 -*-
"""主动性调度器 v1：离线演化 + 主动消息 + 节流阀。

设计：
- tick 由宿主线程周期调用（server 起 daemon 线程；测试用假钟直接调 tick）
- 主动消息条件：会话空闲 ≥ idle_seconds、当日配额未用完、
  上一条发言是玩家（或空闲超过 2 倍阈值——角色可以"忍不住"先开口）
- 节流阀：每会话每日 max_per_day 条，计数存 meta 表，自然日滚动
- 离线演化：空闲 ≥ world_event_min_idle 时，为角色生成一条"离开的这段时间
  世界发生了什么"事实（kind=world）入记忆库，每个自然日至多一次
- 生成走同一 LLM 客户端，输出经 parse_output 剥离尾部块
"""
import json
import time
from contextlib import nullcontext
from datetime import date

from jiuli.parser import parse_output

DEFAULT_CONFIG = {
    "enabled": True,
    "idle_seconds": 6 * 3600,        # 6 小时未说话才主动
    "max_per_day": 3,                # 节流阀：每日每会话上限
    "world_event_min_idle": 24 * 3600,  # 离线 1 天以上产生世界事件
}

PROACTIVE_PROMPT = """你是角色「{name}」。当前状态：{state}
最近剧情末尾：{recent}

你有一段时间没有和玩家说话了。以角色的身份主动发一条消息：
- 不超过两句话，符合人设和当前心情
- 可以是问候、想起某件事、或对最近剧情的后续
- 只输出消息正文，不要任何格式标记或解释"""

WORLD_EVENT_PROMPT = """你是角色「{name}」所处世界的叙述者。
在玩家离开的这段时间里，世界照常运转。写一句话概括角色身边发生的一件小事
（与角色设定相关，第三人称，如「凌夜这几天都在为文化祭的采买跑腿」）。
只输出这一句话。"""


def _parse_day(ts):
    return date.fromtimestamp(ts).isoformat()


class ProactiveEngine:
    def __init__(self, sessions_provider, now_fn=time.time, lock_provider=None):
        """sessions_provider: 返回 RPSession 列表的可调用（server 传注册表快照）。

        lock_provider: 可选 (session) -> 上下文管理器。服务器必须传入每会话
        写锁，否则调度线程的生成/落库会与用户回合竞态（P4 复盘缺陷 #1）。
        """
        self.sessions_provider = sessions_provider
        self.now_fn = now_fn
        self.lock_provider = lock_provider

    # ── 配置（全局，存 meta 表） ──────────────────────────────
    def get_config(self, session):
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(session.store.get_meta("proactive_config", {}))
        return cfg

    def set_config(self, session, patch):
        cfg = self.get_config(session)
        cfg.update({k: v for k, v in patch.items() if k in DEFAULT_CONFIG})
        session.store.set_meta("proactive_config", cfg)
        return cfg

    # ── 主循环 ────────────────────────────────────────────────
    def tick(self):
        """扫描所有已注册会话，触发该触发的。返回事件列表。"""
        events = []
        for s in list(self.sessions_provider()):
            try:
                events.extend(self.tick_session(s))
            except Exception as e:  # noqa: BLE001
                events.append({"session_id": s.session_id, "kind": "error",
                               "message": str(e)})
        return events

    def tick_session(self, s, force=False):
        lock = self.lock_provider(s) if self.lock_provider else nullcontext()
        with lock:
            return self._tick_session_locked(s, force)

    def _tick_session_locked(self, s, force=False):
        cfg = self.get_config(s)
        now = self.now_fn()
        info = s.store.last_message_info(s.session_id)
        if not info:
            return []
        idle = now - info["epoch"]
        events = []

        if cfg.get("enabled") or force:
            if self._should_speak(s, cfg, info, idle, force):
                msg = self._generate(s, PROACTIVE_PROMPT.format(
                    name=s.pkg.persona.get("name", "角色"),
                    state=json.dumps(s.state, ensure_ascii=False),
                    recent=self._recent_tail(s)))
                if msg:
                    s.store.add_message(s.session_id, "assistant", msg, kind="proactive")
                    self._bump_quota(s, now)
                    events.append({"session_id": s.session_id, "kind": "proactive",
                                   "message": msg})

        if self._should_world_event(s, cfg, idle, now):
            fact = self._generate(s, WORLD_EVENT_PROMPT.format(
                name=s.pkg.persona.get("name", "角色")))
            if fact:
                s.memory.add(s.session_id, [{"text": fact, "kind": "world"}])
                s.store.set_meta("world_event:%s:%s" % (s.session_id, _parse_day(now)), fact)
                events.append({"session_id": s.session_id, "kind": "world_event",
                               "message": fact})
        return events

    # ── 条件判断 ──────────────────────────────────────────────
    def _quota_used(self, s, now):
        return s.store.get_meta("proactive_count:" + _parse_day(now), 0)

    def _bump_quota(self, s, now):
        s.store.set_meta("proactive_count:" + _parse_day(now),
                         self._quota_used(s, now) + 1)

    def _should_speak(self, s, cfg, info, idle, force):
        if force:
            return True
        if not cfg.get("enabled"):
            return False
        if idle < cfg["idle_seconds"]:
            return False
        if self._quota_used(s, self.now_fn()) >= cfg["max_per_day"]:
            return False
        # 玩家说完才轮到角色主动；若空闲超 2 倍阈值，角色也可以先开口
        return info["role"] == "user" or idle >= cfg["idle_seconds"] * 2

    def _should_world_event(self, s, cfg, idle, now):
        key = "world_event:%s:%s" % (s.session_id, _parse_day(now))
        return idle >= cfg["world_event_min_idle"] and not s.store.get_meta(key)

    # ── 生成 ──────────────────────────────────────────────────
    @staticmethod
    def _recent_tail(s, chars=200):
        msgs = s.store.all_messages(s.session_id)
        if not msgs:
            return "（对话刚开始）"
        return msgs[-1][2][:chars]

    def _generate(self, s, system):
        raw = s.llm.chat(system, [{"role": "user", "content": "[系统调度：请生成]"}],
                         temperature=0.9, max_tokens=1024)
        narrative, _, _ = parse_output(raw)
        return narrative.strip()
