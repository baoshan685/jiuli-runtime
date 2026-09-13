# -*- coding: utf-8 -*-
"""酒醴 Web 服务器（stdlib ThreadingHTTPServer，零第三方依赖）。

用法：
  python -m jiuli.server --pkg <card_package目录> [--db jiuli.db] --mock
  python -m jiuli.server --pkg <pkg> --api-base URL --api-key K --model M

端点：
  GET  /                          Web UI（单文件 index.html）
  GET  /api/info                  角色与模型配置信息
  GET  /api/sessions              会话列表
  POST /api/sessions              新建会话
  GET  /api/messages?session_id   消息历史
  GET  /api/state?session_id      当前状态
  POST /api/turn                  同步回合 {session_id, input}
  POST /api/turn_stream           SSE 流式回合（同参）
  POST /api/reroll                swipe 重 roll {session_id}
  POST /api/fork                  分叉 {session_id, title?}
  GET  /api/illustrations         插图映射（含本地存在性）
  GET  /api/illustration/file?name=   插图文件
  POST /api/model                 运行时配置模型 {base,key,model}（仅内存）
  POST /api/model/test            连通性测试
  GET  /api/ollama                探测本地 Ollama (127.0.0.1:11434)
  GET  /api/proactive/config      主动性调度配置（含当日配额）
  POST /api/proactive/config      修改配置 {enabled,idle_seconds,max_per_day}
  POST /api/proactive/trigger     强制触发一次主动消息 {session_id}
  GET  /api/proactive/poll        轮询新消息（含主动消息）?session_id&after_id
"""
import argparse
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from jiuli.llm import MockLLM, OpenAICompatClient
from jiuli.loop import RPSession
from jiuli.scheduler import ProactiveEngine
from jiuli.session import SessionStore

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class App:
    """共享状态：多卡片 / llm / store / 会话实例表 + 每会话写锁。"""

    def __init__(self, pkg_dir, llm, db_path, token_budget=4000):
        pkg_dirs = [pkg_dir] if isinstance(pkg_dir, (str, Path)) else list(pkg_dir)
        self.llm = llm
        self.db_path = db_path
        self.token_budget = token_budget
        self._lock = threading.RLock()  # 可重入：session_for_card 会嵌套调 session()
        self._sessions = {}     # session_id -> RPSession
        self._locks = {}        # session_id -> threading.Lock

        # 多卡片：card_id -> {"dir", "pkg"}；CardPackage 含检索器，加载快
        from jiuli.loop import CardPackage
        self.cards = {}
        for d in pkg_dirs:
            cp = CardPackage(d)
            cid = cp.manifest.get("skill_name") or Path(d).name
            self.cards[cid] = {"dir": Path(d), "pkg": cp}
        self.default_card = next(iter(self.cards))

        # 复用每张卡最近会话，避免每次重启都造空会话；无历史才新建
        probe_store = self.cards[self.default_card]["pkg"].dir  # 仅取目录占位
        first = RPSession(self.cards[self.default_card]["dir"], llm, db_path,
                          session_id=self._latest_or_none(self.default_card),
                          token_budget=token_budget, card_id=self.default_card)
        self.default_session_id = first.session_id
        self._sessions[first.session_id] = first
        self.engine = ProactiveEngine(
            lambda: list(self._sessions.values()),
            lock_provider=lambda s: self._locks.setdefault(
                s.session_id, threading.Lock()))

    def _latest_or_none(self, card_id):
        store = self._any_store()
        sid = store.latest_session(card_id)
        return sid

    def _any_store(self):
        """同一 db_path，任意 RPSession 的 store 均可；没有则临时建一个。"""
        if self._sessions:
            return next(iter(self._sessions.values())).store
        return SessionStore(self.db_path)

    def cards_info(self):
        return [{"id": cid,
                 "name": c["pkg"].persona.get("name", cid),
                 "bio": c["pkg"].persona.get("bio", ""),
                 "session_count": len(self._any_store().list_sessions(cid))}
                for cid, c in self.cards.items()]

    def session_for_card(self, card_id=None, sid=None):
        """取某卡片的会话：显式 sid 优先；否则该卡最近会话；再否则新建。"""
        card_id = card_id or self.default_card
        if card_id not in self.cards:
            raise KeyError("unknown card: %s" % card_id)
        with self._lock:
            if sid is not None:
                return self.session(sid)
            existing = self._any_store().latest_session(card_id)
            if existing is not None:
                return self.session(existing)
            s = RPSession(self.cards[card_id]["dir"], self.llm, self.db_path,
                          token_budget=self.token_budget, card_id=card_id)
            self._sessions[s.session_id] = s
            self._locks.setdefault(s.session_id, threading.Lock())
            return s, self._locks[s.session_id]

    def session(self, sid=None):
        with self._lock:
            if sid is None:
                sid = self.default_session_id
            sid = int(sid)
            if sid not in self._sessions:
                # 历史会话必须按它自己的 card_id 找回对应的包（串角色防护）
                store = self._any_store()
                card_id = store.get_session_card(sid) or self.default_card
                if card_id not in self.cards:
                    card_id = self.default_card
                s = RPSession(self.cards[card_id]["dir"], self.llm,
                              self.db_path, session_id=sid,
                              token_budget=self.token_budget)
                self._sessions[sid] = s
            if sid not in self._locks:
                self._locks[sid] = threading.Lock()
            return self._sessions[sid], self._locks[sid]

    def lock_for(self, sid):
        with self._lock:
            sid = int(sid or self.default_session_id)
            if sid not in self._locks:
                self._locks[sid] = threading.Lock()
            return self._locks[sid]


class Handler(BaseHTTPRequestHandler):
    app = None  # 注入

    # ── 基础设施 ──────────────────────────────────────────────
    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    # ── GET ───────────────────────────────────────────────────
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                return self._file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            if u.path == "/api/info":
                p = self.app._sessions[self.app.default_session_id].pkg
                return self._json({"persona": p.persona,
                                   "manifest": p.manifest,
                                   "state": p.state_machine.initial_state()})
            if u.path == "/api/cards":
                return self._json({"cards": self.app.cards_info(),
                                   "default_card": self.app.default_card})
            if u.path == "/api/sessions":
                card_id = (q.get("card_id", [None])[0]
                           or q.get("session_id", [None])[0] and None) or None
                s, _ = self.app.session_for_card(card_id, q.get("session_id", [None])[0])
                return self._json({"sessions": s.store.list_sessions(card_id),
                                   "current": s.session_id})
            if u.path == "/api/messages":
                s, _ = self.app.session(q.get("session_id", [None])[0])
                msgs = s.store.all_messages(s.session_id)
                return self._json({"session_id": s.session_id,
                                   "messages": [{"id": i, "role": r, "content": c, "kind": k} for i, r, c, k in msgs],
                                   "summary": s.store.get_summary(s.session_id)})
            if u.path == "/api/state":
                s, _ = self.app.session(q.get("session_id", [None])[0])
                return self._json({"session_id": s.session_id, "state": s.state})
            if u.path == "/api/illustrations":
                s, _ = self.app.session(q.get("session_id", [None])[0])
                return self._json(self._illustration_map(s))
            if u.path == "/api/illustration/file":
                return self._illu_file(q.get("name", [""])[0])
            if u.path == "/api/ollama":
                return self._json(self._probe_ollama())
            if u.path == "/api/proactive/config":
                s, _ = self.app.session(q.get("session_id", [None])[0])
                return self._json({"config": self.app.engine.get_config(s),
                                   "used_today": s.store.get_meta(
                                       "proactive_count:" + time.strftime("%Y-%m-%d"), 0)})
            if u.path == "/api/proactive/poll":
                s, _ = self.app.session(q.get("session_id", [None])[0])
                after = int(q.get("after_id", [0])[0])
                return self._json({"messages": s.store.messages_after(s.session_id, after),
                                   "state": s.state})
            return self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": str(e)}, 500)

    def _file(self, path, ctype):
        if not path.is_file():
            return self._json({"error": "missing %s" % path.name}, 404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _illustration_map(self, s):
        illu = s.pkg.dir / "illustrations.json"
        data = json.loads(illu.read_text(encoding="utf-8")) if illu.is_file() else {}
        files = []
        base = s.pkg.dir / "assets" / "illustrations"
        if base.is_dir():
            files = sorted(x.name for x in base.iterdir() if x.is_file())
        return {"package": data, "local_files": files}

    def _illu_file(self, name):
        s, _ = self.app.session(None)
        safe = Path(name).name  # 防目录穿越
        p = s.pkg.dir / "assets" / "illustrations" / safe
        if not p.is_file():
            return self._json({"error": "no such illustration"}, 404)
        ctype = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        return self._file(p, ctype)

    @staticmethod
    def _probe_ollama():
        try:
            with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as r:
                tags = json.loads(r.read().decode("utf-8"))
                return {"ok": True, "models": [m.get("name") for m in tags.get("models", [])]}
        except Exception:  # noqa: BLE001
            return {"ok": False, "models": []}

    # ── POST ──────────────────────────────────────────────────
    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._body()
            if u.path == "/api/turn":
                return self._post_turn(body)
            if u.path == "/api/turn_stream":
                return self._post_turn_stream(body)
            if u.path == "/api/sessions":
                card_id = body.get("card_id") or self.app.default_card
                if card_id not in self.app.cards:
                    return self._json({"error": "unknown card"}, 400)
                s, lock = self.app.session_for_card(card_id)
                with lock:
                    sid = s.store.create_session(title=body.get("title", ""),
                                                 card_id=card_id)
                    s.store.set_state(sid, s.pkg.state_machine.initial_state())
                # 新会话立刻挂到运行时实例上，保证后续 turn/poll 命中它
                with self.app._lock:
                    ns = RPSession(self.app.cards[card_id]["dir"], self.app.llm,
                                   self.app.db_path, session_id=sid,
                                   token_budget=self.app.token_budget,
                                   card_id=card_id)
                    self.app._sessions[sid] = ns
                return self._json({"session_id": sid, "card_id": card_id})
            if u.path == "/api/reroll":
                s = self.app._sessions.get(int(body.get("session_id", 0)))
                if not s:
                    return self._json({"error": "bad session"}, 400)
                with self.app.lock_for(s.session_id):
                    r = s.reroll()
                return self._json({"result": r} if r else {"error": "nothing to reroll"}, 200)
            if u.path == "/api/fork":
                s = self.app._sessions.get(int(body.get("session_id", 0)))
                if not s:
                    return self._json({"error": "bad session"}, 400)
                new_id = s.fork(title=body.get("title", ""))
                return self._json({"session_id": new_id})
            if u.path == "/api/model":
                cfg = self.app.llm = OpenAICompatClient(
                    body["base"], body.get("key", ""), body["model"], timeout=240)
                return self._json({"ok": True, "model": cfg.model})
            if u.path == "/api/model/test":
                cfg = self.app.llm
                if isinstance(cfg, MockLLM):
                    return self._json({"ok": True, "reply": "(mock)"})
                try:
                    reply = cfg.chat("你只回复两个字：成功",
                                     [{"role": "user", "content": "ping"}], max_tokens=512)
                except urllib.error.HTTPError as e:
                    hint = "（已尝试自动补 /v1）" if not (cfg.base_url or "").endswith("/v1") else ""
                    return self._json({"ok": False,
                                       "error": "HTTP %d %s %s" % (e.code, e.reason, hint)})
                return self._json({"ok": bool(reply.strip()),
                                   "reply": reply.strip()[:40],
                                   "resolved_base": cfg.base_url})
            if u.path == "/api/proactive/config":
                s, _ = self.app.session(body.get("session_id"))
                cfg = self.app.engine.set_config(s, body)
                return self._json({"ok": True, "config": cfg})
            if u.path == "/api/proactive/trigger":
                s, _ = self.app.session(body.get("session_id"))
                with self.app.lock_for(s.session_id):
                    events = self.app.engine.tick_session(s, force=True)
                return self._json({"events": events})
            return self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": str(e)}, 500)

    @staticmethod
    def _friendly_llm_error(e):
        if isinstance(e, urllib.error.HTTPError):
            if e.code == 402:
                return "HTTP 402: 当前模型额度不足/欠费（免费模型额度每日 0 点刷新）。"                        "请在「模型设置」里换一个模型。"
            if e.code == 401:
                return "HTTP 401: API Key 无效或未授权，请检查「模型设置」。"
            if e.code == 429:
                return "HTTP 429: 触发限流，请稍后重试或降低请求频率。"
            return "HTTP %d: %s" % (e.code, e.reason)
        return str(e)

    def _post_turn(self, body):
        s, lock = self.app.session(body.get("session_id"))
        with lock:
            try:
                r = s.run_turn(body.get("input", ""))
            except urllib.error.HTTPError as e:
                return self._json({"error": self._friendly_llm_error(e)}, 502)
        return self._json(r)

    def _post_turn_stream(self, body):
        s, lock = self.app.session(body.get("session_id"))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        def send(event):
            self.wfile.write(("data: " + json.dumps(event, ensure_ascii=False) + "\n\n")
                             .encode("utf-8"))
            self.wfile.flush()

        with lock:
            try:
                for ev in s.run_turn_stream(body.get("input", "")):
                    send(ev)
            except Exception as e:  # noqa: BLE001
                send({"type": "error", "message": self._friendly_llm_error(e)})
        self.wfile.write(b"data: [DONE]\n\n")


def main():
    ap = argparse.ArgumentParser(prog="jiuli-server")
    ap.add_argument("--pkg", required=True, action="append",
                    help="卡片语义包目录，可重复传入多张卡")
    ap.add_argument("--db", default="jiuli.db")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--tick-seconds", type=int, default=30, dest="tick_seconds")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    if args.mock:
        llm = MockLLM()
    else:
        llm = OpenAICompatClient(args.api_base, args.api_key or "", args.model or "",
                                 timeout=240)
    app = App(args.pkg, llm, args.db, token_budget=args.budget)

    Handler.protocol_version = "HTTP/1.1"
    Handler.app = app
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    # 主动性调度：守护线程周期 tick
    def scheduler_loop():
        while True:
            time.sleep(args.tick_seconds)
            try:
                app.engine.tick()
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=scheduler_loop, daemon=True).start()
    print("酒醴 Web UI: http://%s:%d  (cards=%s, mock=%s, 调度间隔=%ds)"
          % (args.host, args.port, ",".join(app.cards), args.mock,
             args.tick_seconds))
    server.serve_forever()


if __name__ == "__main__":
    main()
