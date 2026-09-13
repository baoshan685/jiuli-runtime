# -*- coding: utf-8 -*-
"""会话存储：SQLite。sessions / messages / state / facts 四张表。

存档是运行时的职责，不再依赖模型自觉写文件。
"""
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT DEFAULT '',
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,            -- user / assistant / system
    content TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state (
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (session_id, key)
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    text TEXT NOT NULL,
    created TEXT NOT NULL,
    kind TEXT DEFAULT 'event',
    weight REAL DEFAULT 1.0,
    active INTEGER DEFAULT 1,
    supersedes INTEGER
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _now_ts():
    return datetime.now().timestamp()


class SessionStore:
    def __init__(self, db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Web 服务器是多线程的：连接允许跨线程使用，读写用 RLock 串行化
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.lock = threading.RLock()
        self.conn.executescript(SCHEMA)
        # 旧库升级
        for stmt in ("ALTER TABLE sessions ADD COLUMN summary TEXT DEFAULT ''",
                     "ALTER TABLE sessions ADD COLUMN card_id TEXT DEFAULT ''",
                     "ALTER TABLE messages ADD COLUMN kind TEXT DEFAULT 'chat'",
                     "ALTER TABLE facts ADD COLUMN kind TEXT DEFAULT 'event'",
                     "ALTER TABLE facts ADD COLUMN weight REAL DEFAULT 1.0",
                     "ALTER TABLE facts ADD COLUMN active INTEGER DEFAULT 1",
                     "ALTER TABLE facts ADD COLUMN supersedes INTEGER"):
            try:
                self.conn.execute(stmt)
            except sqlite3.OperationalError:
                pass
        self.conn.commit()

    def create_session(self, title="", card_id=""):
        with self.lock:
            now = _now()
            cur = self.conn.execute(
                "INSERT INTO sessions (title, created, updated, card_id) "
                "VALUES (?,?,?,?)", (title, now, now, card_id))
            self.conn.commit()
            return cur.lastrowid

    def get_session_card(self, session_id):
        with self.lock:
            row = self.conn.execute("SELECT card_id FROM sessions WHERE id=?",
                                    (session_id,)).fetchone()
        return (row[0] or "") if row else ""

    def latest_session(self, card_id=None):
        """某卡片最近更新的会话 id；card_id 为 None 时不限卡片。无则 None。"""
        with self.lock:
            if card_id is None:
                row = self.conn.execute(
                    "SELECT id FROM sessions ORDER BY updated DESC LIMIT 1").fetchone()
            else:
                row = self.conn.execute(
                    "SELECT id FROM sessions WHERE card_id=? "
                    "ORDER BY updated DESC, id DESC LIMIT 1", (card_id,)).fetchone()
        return row[0] if row else None

    def set_summary(self, session_id, summary):
        with self.lock:
            self.conn.execute("UPDATE sessions SET summary=? WHERE id=?",
                              (summary, session_id))
            self.conn.commit()

    def get_summary(self, session_id):
        with self.lock:
            row = self.conn.execute("SELECT summary FROM sessions WHERE id=?",
                                    (session_id,)).fetchone()
        return row[0] if row and row[0] else ""

    def count_messages(self, session_id):
        with self.lock:
            return self.conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id=?",
                (session_id,)).fetchone()[0]

    def delete_last_exchange(self, session_id):
        """删除最后一组 (assistant, user) 消息，供 swipe 重 roll 使用。

        返回被删的用户输入文本（若无完整交换则返回 None）。
        """
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, role, content FROM messages WHERE session_id=? "
                "ORDER BY id DESC LIMIT 2", (session_id,)).fetchall()
            if not rows:
                return None
            removed_user = None
            ids = []
            if rows[0][1] == "assistant" and len(rows) == 2 and rows[1][1] == "user":
                ids = [rows[0][0], rows[1][0]]
                removed_user = rows[1][2]
            elif rows[0][1] == "assistant":
                ids = [rows[0][0]]
            elif rows[0][1] == "user":
                ids = [rows[0][0]]
                removed_user = rows[0][2]
            for i in ids:
                self.conn.execute("DELETE FROM messages WHERE id=?", (i,))
            self.conn.commit()
        return removed_user

    def fork_session(self, src_session_id, upto_message_id=None, title=""):
        """从源会话复制全部（或截止到某条消息的）历史到新会话，状态一并复制。

        记忆库（facts）是全局的，不随分叉复制。
        """
        new_id = self.create_session(title=title or ("fork #%d" % src_session_id))
        with self.lock:
            if upto_message_id is None:
                rows = self.conn.execute(
                    "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
                    (src_session_id,)).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT role, content FROM messages WHERE session_id=? AND id<=? "
                    "ORDER BY id", (src_session_id, upto_message_id)).fetchall()
        for role, content in rows:
            self.add_message(new_id, role, content)
        state = self.get_state(src_session_id)
        if state:
            self.set_state(new_id, state)
        # 摘要也是会话上下文的一部分（P4 复盘缺陷 #3），一并复制
        summary = self.get_summary(src_session_id)
        if summary:
            self.set_summary(new_id, summary)
        return new_id

    def add_message(self, session_id, role, content, kind="chat"):
        with self.lock:
            self.conn.execute(
                "INSERT INTO messages (session_id, role, content, created, kind) "
                "VALUES (?,?,?,?,?)", (session_id, role, content, _now(), kind))
            self.conn.execute("UPDATE sessions SET updated=? WHERE id=?", (_now(), session_id))
            self.conn.commit()

    def last_message_info(self, session_id):
        """返回 (id, role, created, epoch) 或 None——供主动性调度器判断空闲。"""
        with self.lock:
            row = self.conn.execute(
                "SELECT id, role, created FROM messages WHERE session_id=? "
                "ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
        if not row:
            return None
        try:
            ts = datetime.fromisoformat(row[2]).timestamp()
        except ValueError:
            ts = 0.0
        return {"id": row[0], "role": row[1], "created": row[2], "epoch": ts}

    def messages_after(self, session_id, after_id):
        """返回 id > after_id 的消息（含主动消息），供 UI 轮询。"""
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, role, content, kind FROM messages WHERE session_id=? "
                "AND id>? ORDER BY id", (session_id, after_id)).fetchall()
        return [{"id": r[0], "role": r[1], "content": r[2], "kind": r[3]} for r in rows]

    def get_meta(self, key, default=None):
        with self.lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                              (key, json.dumps(value, ensure_ascii=False)))
            self.conn.commit()

    def recent_messages(self, session_id, limit=12):
        """返回最近 limit 条（时间正序），元素为 (id, role, content, kind)。"""
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, role, content, kind FROM messages WHERE session_id=? "
                "ORDER BY id DESC LIMIT ?", (session_id, limit)).fetchall()
        return list(reversed(rows))

    def all_messages(self, session_id):
        with self.lock:
            return self.conn.execute(
                "SELECT id, role, content, kind FROM messages WHERE session_id=? "
                "ORDER BY id", (session_id,)).fetchall()

    def get_state(self, session_id):
        with self.lock:
            rows = self.conn.execute(
                "SELECT key, value FROM state WHERE session_id=?", (session_id,)).fetchall()
        return {k: json.loads(v) for k, v in rows}

    def set_state(self, session_id, state):
        with self.lock:
            for k, v in state.items():
                self.conn.execute(
                    "INSERT OR REPLACE INTO state (session_id, key, value) VALUES (?,?,?)",
                    (session_id, k, json.dumps(v, ensure_ascii=False)))
            self.conn.commit()

    def add_fact(self, session_id, text, kind="event", weight=1.0):
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO facts (session_id, text, created, kind, weight, active) "
                "VALUES (?,?,?,?,?,1)", (session_id, text, _now(), kind, weight))
            self.conn.commit()
            return cur.lastrowid

    def add_facts(self, session_id, texts, kind="event"):
        for t in texts:
            if (t or "").strip():
                self.add_fact(session_id, t.strip(), kind=kind)

    def active_facts(self):
        """全部生效事实（全会话共享记忆库），含元数据供记忆引擎决策。"""
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, text, kind, weight, created FROM facts WHERE active=1 "
                "ORDER BY id").fetchall()
        return [dict(id=r[0], text=r[1], kind=r[2], weight=r[3], created=r[4])
                for r in rows]

    def facts_version(self):
        """生效事实的轻量版本号 (count, max_id)——供多会话缓存失效检测。"""
        with self.lock:
            row = self.conn.execute(
                "SELECT COUNT(*), IFNULL(MAX(id), 0) FROM facts WHERE active=1"
            ).fetchone()
        return (row[0], row[1])

    def deactivate_fact(self, fact_id, superseded_by=None):
        with self.lock:
            self.conn.execute("UPDATE facts SET active=0, supersedes=? WHERE id=?",
                              (superseded_by, fact_id))
            self.conn.commit()

    def retract_facts(self, fact_ids):
        """撤回一组事实（swipe 重roll 用）：停用事实本身，并恢复被它们
        推翻的旧事实。返回 (停用数, 恢复数)。"""
        fids = [int(i) for i in fact_ids if i]
        if not fids:
            return (0, 0)
        with self.lock:
            marks = ",".join("?" * len(fids))
            n1 = self.conn.execute(
                "UPDATE facts SET active=0 WHERE id IN (%s)" % marks,
                fids).rowcount
            n2 = self.conn.execute(
                "UPDATE facts SET active=1, supersedes=NULL WHERE active=0 "
                "AND supersedes IN (%s)" % marks, fids).rowcount
            self.conn.commit()
        return (n1, n2)

    def prune_facts(self, keep):
        """遗忘修剪：只保留最近 keep 条生效事实。返回停用条数。"""
        with self.lock:
            n = self.conn.execute("SELECT COUNT(*) FROM facts WHERE active=1").fetchone()[0]
            if n <= keep:
                return 0
            cutoff = self.conn.execute(
                "SELECT id FROM facts WHERE active=1 ORDER BY id DESC LIMIT 1 OFFSET ?",
                (keep - 1,)).fetchone()
            self.conn.execute("UPDATE facts SET active=0 WHERE active=1 AND id<=?",
                              (cutoff[0],))
            self.conn.commit()
            return n - keep

    def all_facts(self, session_id=None):
        with self.lock:
            if session_id is None:
                rows = self.conn.execute("SELECT text FROM facts").fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT text FROM facts WHERE session_id=?", (session_id,)).fetchall()
        return [r[0] for r in rows]

    def list_sessions(self, card_id=None):
        with self.lock:
            if card_id is None:
                return self.conn.execute(
                    "SELECT id, title, created, updated FROM sessions "
                    "ORDER BY updated DESC").fetchall()
            return self.conn.execute(
                "SELECT id, title, created, updated FROM sessions WHERE card_id=? "
                "ORDER BY updated DESC", (card_id,)).fetchall()

    def close(self):
        with self.lock:
            self.conn.close()
