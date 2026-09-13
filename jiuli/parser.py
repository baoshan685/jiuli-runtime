# -*- coding: utf-8 -*-
"""模型输出解析：叙事正文 + 可选结构化尾部（```jiuli {json}```）。

解析失败不致命：降级为纯叙事文本，由调用方决定重试或告警。
"""
import json
import re

TAIL_RE = re.compile(r"```jiuli\s*(\{.*?\})\s*```", re.DOTALL)
ALLOWED_KEYS = {"state_diff", "illustration", "memory_facts", "summary", "suggestions"}


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
