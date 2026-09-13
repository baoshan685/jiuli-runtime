# -*- coding: utf-8 -*-
"""LLM 接入层：OpenAI 兼容（urllib，零依赖）+ 离线 MockLLM。

RP 每轮只调用一次模型（生成 + 结构化尾部同输出），
不做多步 agent 循环——这是瘦上下文成本承诺的架构前提。
"""
import json
import urllib.error
import urllib.request


class OpenAICompatClient:
    def __init__(self, base_url, api_key, model, timeout=120):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # base_url 允许三种写法：host、host/v1、host/v1/（见 _endpoint 的自动纠正）
        self.base_url = base_url.strip().rstrip("/") if base_url else ""

    def _endpoint(self, path):
        """拼端点；用户漏写 /v1 时自动补上（记录是否已纠正，供提示）。

        判据：404 且 base 尾段不是版本号（v1/v2/beta等）→ 补 /v1 重试。
        这里先生成候选端点序列。
        """
        bases = [self.base_url]
        tail = self.base_url.rsplit("/", 1)[-1].lower()
        if tail not in ("v1", "v2", "v3", "beta", "openai"):
            bases.append(self.base_url + "/v1")
        return [b + path for b in bases]

    def _post(self, path, payload, stream=False):
        """POST，带 /v1 自动纠正。返回 (resp, resolved_base)。"""
        data = json.dumps(payload).encode("utf-8")
        last_err = None
        for base in [self.base_url] + (
                [self.base_url + "/v1"]
                if self.base_url.rsplit("/", 1)[-1].lower()
                not in ("v1", "v2", "v3", "beta", "openai") else []):
            req = urllib.request.Request(
                base + path, data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + self.api_key})
            try:
                resp = urllib.request.urlopen(req, timeout=self.timeout)
                if base != self.base_url:
                    self.base_url = base  # 记住纠正结果
                return resp
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    last_err = e
                    continue  # 试下一个 base 候选
                raise
        raise last_err

    def chat(self, system, messages, temperature=0.8, max_tokens=4000):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        with self._post("/chat/completions", payload) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        self.last_usage = data.get("usage")
        return data["choices"][0]["message"]["content"]

    def chat_stream(self, system, messages, temperature=0.8, max_tokens=4000):
        """SSE 流式；逐步 yield 正文增量。"""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        self.last_usage = None
        with self._post("/chat/completions", payload) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if usage:
                    self.last_usage = usage
                choices = chunk.get("choices") or [{}]
                piece = (choices[0].get("delta") or {}).get("content") or ""
                if piece:
                    yield piece


class MockLLM:
    """离线确定性引擎：用于测试与无 API 演示。

    行为：复述玩家输入为叙事，附带结构化尾部——
    存在 affection 变量时 +1，并产出一条记忆事实。
    """

    def __init__(self, state_var="affection"):
        self.state_var = state_var

    def chat(self, system, messages, temperature=0.8, max_tokens=2000):
        last_user = next((m["content"] for m in reversed(messages)
                          if m["role"] == "user"), "")
        tail = {"memory_facts": ["玩家曾说：" + last_user[:40]], "summary": "演示模式剧情"}
        if self.state_var:
            tail["state_diff"] = {self.state_var: "+1"}
        narrative = ("（演示模式）她看着你，轻声回应了「%s」。炉火噼啪作响，"
                     "这一刻安静得像被折叠进了书页之间。" % last_user[:30])
        self.last_usage = {"prompt_tokens": 500, "completion_tokens": 100}
        return narrative + "\n```jiuli\n" + json.dumps(tail, ensure_ascii=False) + "\n```"

    def chat_stream(self, system, messages, temperature=0.8, max_tokens=2000):
        text = self.chat(system, messages, temperature, max_tokens)
        for i in range(0, len(text), 6):
            yield text[i:i + 6]
