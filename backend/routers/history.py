"""
历史记录路由（需登录）：按数据集分组列出分析包，并支持删除。

- GET  /api/history/datasets        当前用户全部数据集（按创建时间倒序）
- GET  /api/history/packages?dataset_id= 某数据集下的分析包列表
- DELETE /api/history/dataset/{dataset_id}   删除数据集及其分析包（级联）
- DELETE /api/history/package/{package_id}   删除单个分析包
"""
import logging
import os
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.db import crud
from backend.services.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/history", tags=["历史记录"])


# ---------- P2：收藏 / 分组 ----------
class ToggleFavoriteReq(BaseModel):
    package_id: str
    starred: bool = True


class FavoriteMetaReq(BaseModel):
    package_id: str
    display_name: str | None = None
    group_name: str | None = None
    sort_order: int | None = None


class CreateShareReq(BaseModel):
    package_id: str
    expire_at: float | None = None  # 过期时间戳（秒），可空表示永久


@router.get("/datasets")
def get_datasets(user: dict = Depends(get_current_user)):
    datasets = crud.list_datasets_by_user(user["id"])
    # 统计每个数据集下的分析包数量（便于前端展示）
    result = []
    for ds in datasets:
        count = len(crud.list_packages_by_user(user["id"], ds.get("dataset_id", "")))
        meta = dict(ds)
        meta["package_count"] = count
        result.append(meta)
    return {"datasets": result}


@router.get("/packages")
def get_packages(dataset_id: str, user: dict = Depends(get_current_user)):
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id 不能为空")
    packages = crud.list_packages_by_user(user["id"], dataset_id)
    return {"packages": packages}


@router.delete("/dataset/{dataset_id}")
def delete_dataset(dataset_id: str, user: dict = Depends(get_current_user)):
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id 不能为空")
    paths = crud.delete_dataset_by_user(user["id"], dataset_id)
    # 物理删 pkl（失败仅告警，不阻断）
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception as e:
            logger.warning("删除历史 pkl 失败（已忽略）: %s -> %s", p, e)
    return {"success": True}


@router.delete("/package/{package_id}")
def delete_package(package_id: str, user: dict = Depends(get_current_user)):
    if not package_id:
        raise HTTPException(status_code=400, detail="package_id 不能为空")
    ok = crud.delete_package_by_user(user["id"], package_id)
    if not ok:
        raise HTTPException(status_code=404, detail="分析包不存在或无权删除")
    return {"success": True}


@router.delete("/sessions/{session_id}")
def delete_history_session(session_id: str, user: dict = Depends(get_current_user)):
    """删除指定历史会话（仅限本人会话）。

    - 归属校验：仅删除 user_id == 当前登录用户的会话；不属于自己的返回 404
      （与 restore_session 行为一致：不暴露会话是否存在）。
    - 级联清理：sessions / datasets / analysis_packages 三表 + 数据集 pkl 文件一并清理。
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    ok, pkl_paths = crud.delete_session_by_user(user["id"], session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="历史会话不存在或无权删除")
    # 物理删除 pkl（失败仅告警，不阻断）
    for p in pkl_paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception as e:
            logger.warning("删除会话 pkl 失败（已忽略）: %s -> %s", p, e)
    return {"success": True}


@router.get("/favorites")
def get_favorites(user: dict = Depends(get_current_user)):
    """当前用户的收藏列表（按分组 + 排序聚合）。"""
    groups = crud.list_favorites_by_user(user["id"])
    return {"groups": groups}


# ---------- P3：会话维度历史（智能体演进基础） ----------
@router.get("/sessions")
def get_sessions(user: dict = Depends(get_current_user)):
    """返回当前用户的全部历史会话摘要。

    每条含 session_id / title / last_page / dataset_count / package_count /
    created_at / last_access，按最后访问倒序。前端据此渲染"会话历史"列表。
    """
    sessions = crud.list_sessions_by_user(user["id"])
    try:
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from backend.routers._sanitize import clean_for_json
        sessions = clean_for_json(sessions)
    except Exception:
        pass
    return {"sessions": sessions}


@router.post("/sessions/{session_id}/restore")
def restore_session(session_id: str, user: dict = Depends(get_current_user)):
    """恢复指定历史会话：仅当该会话归属当前用户时允许。

    返回该会话的 session_id 与完整轻量状态，前端将本地会话切换至此 session
    （替换 localStorage sessionId 并用 state 重建上下文），实现"点进去回到上次分析"。
    返回最后一个访问页面 last_page，供前端智能路由跳转。
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    # 校验归属
    rows = crud.list_sessions_by_user(user["id"])
    if not any(s["session_id"] == session_id for s in rows):
        raise HTTPException(status_code=404, detail="历史会话不存在或无权访问")
    state = crud.load_session_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="会话数据已丢失")
    # 点击进入即视为"最近操作"：刷新 last_access，使该会话在裁剪时按"最后一次使用"而非创建时间排序。
    # 例：7/1 上传、7/3 从历史点进 → 最后访问时间变为 7/3，不会被当作"最旧"清掉。
    crud.touch_session(session_id, time.time())
    # 兜底：state 里若残留 NaN/Inf（历史脏数据），前端 json 序列化会再炸一次，这里静默净化。
    try:
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from backend.routers._sanitize import clean_for_json
        state = clean_for_json(state)
    except Exception:
        pass
    return {
        "session_id": session_id,
        "state": state,
        "last_page": state.get("last_page") or "upload",
    }


@router.post("/favorites/toggle")
def toggle_favorite(req: ToggleFavoriteReq, user: dict = Depends(get_current_user)):
    """收藏 / 取消收藏某分析包。"""
    if not req.package_id:
        raise HTTPException(status_code=400, detail="package_id 不能为空")
    state = crud.toggle_favorite(user["id"], req.package_id, req.starred)
    return {"success": True, "state": state}


@router.post("/favorites/meta")
def update_favorite_meta(req: FavoriteMetaReq, user: dict = Depends(get_current_user)):
    """更新收藏项的显示名 / 分组 / 排序。"""
    if not req.package_id:
        raise HTTPException(status_code=400, detail="package_id 不能为空")
    try:
        state = crud.set_favorite_meta(
            user["id"], req.package_id,
            display_name=req.display_name,
            group_name=req.group_name,
            sort_order=req.sort_order,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="尚未收藏该分析包")
    return {"success": True, "state": state}


# ---------- P2：分享链接 ----------
@router.post("/shares")
def create_share(req: CreateShareReq, user: dict = Depends(get_current_user)):
    """为某分析包生成公开只读分享链接。"""
    if not req.package_id:
        raise HTTPException(status_code=400, detail="package_id 不能为空")
    share = crud.create_share(user["id"], req.package_id, req.expire_at)
    return {"success": True, **share}


@router.get("/shares")
def get_my_shares(user: dict = Depends(get_current_user)):
    """当前用户创建的全部分享。"""
    return {"shares": crud.list_shares_by_user(user["id"])}


@router.delete("/share/{share_id}")
def delete_share(share_id: str, user: dict = Depends(get_current_user)):
    """取消某分享（仅分享者本人）。"""
    if not share_id:
        raise HTTPException(status_code=400, detail="share_id 不能为空")
    ok = crud.delete_share_by_user(user["id"], share_id)
    if not ok:
        raise HTTPException(status_code=404, detail="分享不存在或无权删除")
    return {"success": True}


@router.get("/shared/{share_id}")
def get_shared_package(share_id: str):
    """公开只读：无需登录即可读取被分享的分析包内容。过期/不存在返回 404。"""
    if not share_id:
        raise HTTPException(status_code=400, detail="share_id 不能为空")
    share = crud.get_share(share_id)
    if share is None:
        raise HTTPException(status_code=404, detail="分享不存在或已过期")
    return {
        "share_id": share["share_id"],
        "package_id": share["package_id"],
        "created_at": share["created_at"],
        "expire_at": share["expire_at"],
        "payload": share["payload"],
    }
