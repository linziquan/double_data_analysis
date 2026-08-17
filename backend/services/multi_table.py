"""多表协作工具（合并宽表懒探测）

供 backend/routers/upload.py 与 backend/routers/chat.py 复用，避免循环 import。
"""
from __future__ import annotations

import logging
from typing import Any

_LOGGER = logging.getLogger("uvicorn.error")


def maybe_auto_merge(manager, session_id: str) -> dict[str, Any]:
    """多表懒自动合并：检测 session 是否有 ≥2 张非合并数据集且尚无 merged 宽表。

    若是，则识别关联键链式合并，并把结果注册成 is_merged 数据集（不抢占 active）。
    静默失败：无关联键 / 合并异常时直接跳过，不抛异常给上层。

    返回结构化结果供调用方日志或前端展示：
        - {"status": "skip", "reason": "..."}  无可合并项
        - {"status": "merged", "dataset_id": ..., "rows": ..., "columns": ...}
    """
    session = manager.get_session(session_id)
    if session is None:
        return {"status": "skip", "reason": "session_not_found"}

    datasets = session.datasets or {}
    non_merged = [
        did for did, ds in datasets.items()
        if not getattr(ds, "is_merged", False)
    ]
    if len(non_merged) < 2:
        return {"status": "skip", "reason": "less_than_two_non_merged"}
    if any(getattr(ds, "is_merged", False) for ds in datasets.values()):
        return {"status": "skip", "reason": "already_has_merged"}

    pairs = [(did, manager.get_dataset_df(session_id, did)) for did in non_merged]
    valid = [(did, df) for did, df in pairs if df is not None and not df.empty]
    if len(valid) < 2:
        return {"status": "skip", "reason": "no_valid_dfs"}

    try:
        from src.merge.dataset_merger import build_analysis_units
    except Exception as e:
        _LOGGER.warning(f"[maybe_auto_merge] import failed: {e}")
        return {"status": "skip", "reason": f"import_error:{type(e).__name__}"}

    file_names = {did: (session.datasets[did].file_name or did) for did, _ in valid}
    try:
        units = build_analysis_units(valid, file_names=file_names, llm_cfg=None)
    except Exception as e:
        _LOGGER.warning(f"[maybe_auto_merge] build_analysis_units failed: {e}")
        return {"status": "skip", "reason": f"merge_error:{type(e).__name__}"}

    merged_unit = next((u for u in units if u.kind == "merged"), None)
    if merged_unit is None or merged_unit.df is None:
        return {"status": "skip", "reason": "no_joinable_keys"}

    try:
        did = manager.add_merged_dataset(
            session_id, merged_unit.df,
            sources=merged_unit.sources, keys=merged_unit.keys,
            file_name="合并宽表",
        )
    except Exception as e:
        _LOGGER.warning(f"[maybe_auto_merge] add_merged_dataset failed: {e}")
        return {"status": "skip", "reason": f"register_error:{type(e).__name__}"}

    return {
        "status": "merged",
        "dataset_id": did,
        "rows": int(merged_unit.df.shape[0]),
        "columns": list(merged_unit.df.columns),
        "keys": list(merged_unit.keys),
        "sources": list(merged_unit.sources),
    }
