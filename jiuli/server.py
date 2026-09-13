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

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class App:
    """共享状态：pkg/llm/store/会话实例表 + 每会话写锁。"""

    def __init__(self, pkg_dir, llm, db_path, token_budget=4000):
        self.pkg_dir = pkg_dir
        self.llm = llm
        self.db_path = db_path
        self.token_budget = token_budget
        self._lock = threading.Lock()
        self._sessions = {}     # session_id -> RPSession
        self._locks = {}        # session_id -> threading.Lock
        probe = RPSession(pkg_dir, llm, db_path, token_budget=token_budget)
        self.default_session_id = probe.session_id
        self._sessions[probe.session_id] = probe
        self.engine = ProactiveEngine(
            lambda: list(self._sessions.values()),
            lock_provider=lambda s: self._locks.setdefault(
                s.session_id, threading.Lock()))

    def session(self, sid=None):
        with self._lock:
            if sid is None:
                sid = self.default_session_id
            sid = int(sid)
            if sid not in self._sessions:
                s = RPSession(self.pkg_dir, self.llm, self.db_path,
                              session_id=sid, token_budget=self.token_budget)
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
            if u.path == "/api/sessions":
                s, _ = self.app.session(q.get("session_id", [None])[0])
                return self._json({"sessions": s.store.list_sessions(),
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
                s, _ = self.app.session(None)
                sid = s.store.create_session(title=body.get("title", ""))
                s.store.set_state(sid, s.pkg.state_machine.initial_state())
                return self._json({"session_id": sid})
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
                reply = cfg.chat("你只回复两个字：成功",
                                 [{"role": "user", "content": "ping"}], max_tokens=512)
                return self._json({"ok": bool(reply.strip()), "reply": reply.strip()[:40]})
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

    def _post_turn(self, body):
        s, lock = self.app.session(body.get("session_id"))
        with lock:
            r = s.run_turn(body.get("input", ""))
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
                send({"type": "error", "message": str(e)})
        self.wfile.write(b"data: [DONE]\n\n")


def main():
    ap = argparse.ArgumentParser(prog="jiuli-server")
    ap.add_argument("--pkg", required=True)
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
    print("酒醴 Web UI: http://%s:%d  (pkg=%s, mock=%s, 调度间隔=%ds)"
          % (args.host, args.port, args.pkg, args.mock, args.tick_seconds))
    server.serve_forever()


if __name__ == "__main__":
    main()
