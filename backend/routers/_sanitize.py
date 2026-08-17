"""内部工具：兜底净化历史脏 state / payload 里的 NaN/Inf。"""
from typing import Any
import math


def clean_for_json(obj: Any) -> Any:
    """递归把 NaN / +Inf / -Inf 替换为 None，保证下游 JSON 序列化不再炸。"""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, set):
        return [clean_for_json(v) for v in obj]
    return obj
