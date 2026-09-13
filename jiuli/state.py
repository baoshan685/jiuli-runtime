# -*- coding: utf-8 -*-
"""状态机：变量定义加载、模型输出 diff 的校验与落库。

原则：数值状态是数据库字段，不是模型的自觉。模型只能提出 diff，
合法性由这里裁决——范围外、类型错、未知变量一律拒绝并告警。
"""
import json


class StateMachine:
    def __init__(self, state_machine_json, enabled=True):
        sm = json.loads(state_machine_json) if isinstance(state_machine_json, str) \
            else state_machine_json
        self.variables = {}
        if enabled:
            for v in sm.get("variables", []):
                # 关键词低置信提示默认不启用，需显式 explicit 或运行时配置放行
                if v.get("explicit") or v.get("enabled_hint"):
                    self.variables[v["name"]] = {
                        "label": v.get("label", v["name"]),
                        "default": v.get("default", 0),
                        "min": v.get("min"),
                        "max": v.get("max"),
                        "max_len": v.get("max_len"),
                    }

    def initial_state(self):
        return {name: v["default"] for name, v in self.variables.items()}

    def validate_and_apply(self, state, diff):
        """对模型提出的 state_diff 做校验，返回 (new_state, applied, rejected)。

        diff 允许两种形式：{"affection": 5}（覆盖值）或 {"affection": "+5"}（增量）。
        """
        applied, rejected = {}, []
        if not isinstance(diff, dict):
            return dict(state), applied, ["diff 不是对象"]
        for key, val in diff.items():
            if key not in self.variables:
                rejected.append({"key": key, "reason": "未知变量"})
                continue
            var = self.variables[key]
            cur = state.get(key, var["default"])
            delta = False
            if isinstance(val, str) and val[:1] in "+-":
                delta = True
                try:
                    num = float(val)
                except ValueError:
                    rejected.append({"key": key, "value": val, "reason": "非法增量"})
                    continue
                new = cur + num
            elif isinstance(val, (int, float)):
                new = val
            else:
                # 字符串型变量（如 mood）直接覆盖，可带 max_len 约束：
                # 状态是字段，不是自由文本——整段心理描写不合格
                s_val = str(val)
                ml = var.get("max_len")
                if ml and len(s_val) > int(ml):
                    rejected.append({"key": key, "value": val,
                                     "reason": "字符串超长（>%d 字）" % int(ml)})
                    continue
                applied[key] = s_val
                continue
            if var.get("min") is not None and new < var["min"]:
                rejected.append({"key": key, "value": new, "reason": "低于下限"})
                continue
            if var.get("max") is not None and new > var["max"]:
                rejected.append({"key": key, "value": new, "reason": "超过上限"})
                continue
            if isinstance(cur, int) and isinstance(new, float) and new.is_integer():
                new = int(new)
            applied[key] = new
        new_state = dict(state)
        new_state.update(applied)
        return new_state, applied, rejected
