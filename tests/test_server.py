# -*- coding: utf-8 -*-
"""HTTP 层冒烟测试：起真实 ThreadingHTTPServer（MockLLM），全端点走网络栈。

运行：python -m unittest discover tests（与运行时测试一起执行）
"""
import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jiuli.llm import MockLLM  # noqa: E402
from jiuli.server import App, Handler  # noqa: E402
from test_runtime import make_pkg  # noqa: E402  # 复用同一套 fixture


def api_call(port, method, path, body=None, timeout=30):
    url = "http://127.0.0.1:%d%s" % (port, path)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


class ServerFixture(unittest.TestCase):
    """起一个隔离的 mock 服务器，测试完关闭。"""

    def start_server(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        app = App(make_pkg(self.tmp), MockLLM(),
                  str(Path(self.tmp) / "web.db"), token_budget=4000)

        class H(Handler):
            pass
        H.app = app
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop_server(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestServerEndpoints(ServerFixture):
    def setUp(self):
        self.start_server()

    def tearDown(self):
        self.stop_server()

    def test_index_served(self):
        status, _body, ctype = self._get_raw("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)

    def _get_raw(self, path):
        with urllib.request.urlopen("http://127.0.0.1:%d%s" % (self.port, path)) as r:
            return r.status, r.read().decode("utf-8"), r.headers.get("Content-Type", "")

    def test_info_and_sessions(self):
        _, info = api_call(self.port, "GET", "/api/info")
        self.assertIn("persona", info)
        _, sess = api_call(self.port, "GET", "/api/sessions")
        self.assertEqual(sess["current"], 1)

    def test_turn_messages_reroll_fork(self):
        _, r = api_call(self.port, "POST", "/api/turn",
                        {"session_id": 1, "input": "我们去旧书店逛逛"})
        self.assertIn("narrative", r)
        self.assertEqual(r["state"]["affection"], 1)
        _, msgs = api_call(self.port, "GET", "/api/messages?session_id=1")
        self.assertEqual(len(msgs["messages"]), 2)
        self.assertIn("kind", msgs["messages"][0])
        _, rr = api_call(self.port, "POST", "/api/reroll", {"session_id": 1})
        self.assertIn("result", rr)
        _, fk = api_call(self.port, "POST", "/api/fork", {"session_id": 1})
        self.assertIn("session_id", fk)

    def test_state_endpoint(self):
        _, st = api_call(self.port, "GET", "/api/state?session_id=1")
        self.assertIn("affection", st["state"])

    def test_proactive_config_roundtrip(self):
        _, cfg = api_call(self.port, "GET", "/api/proactive/config")
        self.assertIn("max_per_day", cfg["config"])
        _, saved = api_call(self.port, "POST", "/api/proactive/config",
                            {"idle_seconds": 60, "max_per_day": 5})
        self.assertEqual(saved["config"]["idle_seconds"], 60)
        self.assertEqual(saved["config"]["max_per_day"], 5)

    def test_proactive_poll(self):
        api_call(self.port, "POST", "/api/turn", {"session_id": 1, "input": "hi"})
        _, poll = api_call(self.port, "GET", "/api/proactive/poll?session_id=1&after_id=0")
        self.assertGreaterEqual(len(poll["messages"]), 2)

    def test_illustrations_endpoint(self):
        _, data = api_call(self.port, "GET", "/api/illustrations")
        self.assertIn("package", data)
        self.assertIn("local_files", data)

    def test_404(self):
        try:
            api_call(self.port, "GET", "/api/nope")
            self.fail("expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_turn_stream_sse(self):
        req = urllib.request.Request(
            "http://127.0.0.1:%d/api/turn_stream" % self.port,
            data=json.dumps({"session_id": 1, "input": "流式"}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        deltas, saw_done = 0, False
        with urllib.request.urlopen(req, timeout=30) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    saw_done = True
                    break
                ev = json.loads(payload)
                if ev["type"] == "delta":
                    deltas += 1
        self.assertGreater(deltas, 0)
        self.assertTrue(saw_done)


class TestServerIsolation(ServerFixture):
    """需要独立实例（会替换全局 llm）的端点。"""

    def setUp(self):
        self.start_server()

    def tearDown(self):
        self.stop_server()

    def test_model_test_endpoint(self):
        _, r = api_call(self.port, "POST", "/api/model/test", {})
        self.assertTrue(r["ok"])  # mock 模式直接返回 ok


if __name__ == "__main__":
    unittest.main()
