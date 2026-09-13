# -*- coding: utf-8 -*-
"""模型输出解析：叙事正文 + 可选结构化尾部（```jiuli {json}```）。

解析失败不致命：降级为纯叙事文本，由调用方决定重试或告警。
"""
import json
import re

TAIL_RE = re.compile(r"```jiuli\s*(\{.*?\})\s*```", re.DOTALL)
ALLOWED_KEYS = {"state_diff", "illustration", "memory_facts", "summary", "suggestions"}

# 卡片自带的"叙事状态栏"（酒馆卡常见约定）+ 本运行时约定的 ```status 围栏
STATUS_FENCE_RE = re.compile(r"```status\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
STATUS_TAG_RE = re.compile(
    r"<(userStatusBlocks|statusblock|statusbar)>(.*?)</\1\s*>",
    re.DOTALL | re.IGNORECASE)


def extract_statusbar(narrative):
    """从叙事中抽取状态栏块，返回 (clean_narrative, statusbar_or_None)。

    支持两种形态：```status ...``` 围栏、<userStatusBlocks>/<statusblock>/
    <statusbar> 标签（酒馆卡经正则渲染的等价物）。抽取后正文不再重复展示，
    由 UI 渲染到右侧状态栏面板——相当于接管了卡内正则脚本的职责。
    """
    text = narrative or ""
    m = STATUS_TAG_RE.search(text) or STATUS_FENCE_RE.search(text)
    if not m:
        return text, None
    body = (m.group(2) if m.lastindex >= 2 else m.group(1)).strip()
    if not body:
        return text, None
    clean = (text[:m.start()] + text[m.end():]).strip()
    return clean, body


def parse_output(text):
    """返回 (narrative, tail_dict_or_None, warnings)。"""
    warnings = []
    m = None
    for m in TAIL_RE.finditer(text or ""):
        pass
    if not m:
        return (text or "").strip(), None, warnings
    narrative = (text[:m.start()] + text[m.end():]).strip()
    try:
        tail = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        warnings.append("尾部 JSON 解析失败: %s" % e)
        return narrative, None, warnings
    if not isinstance(tail, dict):
        warnings.append("尾部不是对象")
        return narrative, None, warnings
    unexpected = set(tail) - ALLOWED_KEYS
    if unexpected:
        warnings.append("忽略未知尾部字段: %s" % ",".join(sorted(unexpected)))
        for k in unexpected:
            tail.pop(k, None)
    return narrative, tail or None, warnings
