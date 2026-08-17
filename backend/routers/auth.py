"""
认证路由：注册 / 登录（带游客 session 回填）/ 个人信息 / 改密 / 退出。

关键安全设计：
- 注册即签发 token（建议 2：注册成功自动登录）。
- 登录回填：接收前端游客 session_id，仅当该 session 满足「存在 + user_id IS NULL + 30 分钟内活跃」
  才把 user_id 回填到登录用户，杜绝越权绑定（Bug 2）。
- 改密 / 退出都走 revoke（token_version += 1），旧 token 立即失效（Bug 4）。
"""
import re
import logging
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from backend.db import crud
from backend.services.auth import (
    create_refresh_token,
    create_token,
    get_current_user,
    hash_password,
    refresh_access_token,
    verify_password,
)
from backend.services.session_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_\u4e00-\u9fa5]{3,32}$")


class RegisterRequest:
    pass


@router.post("/register")
def register(
    username: str = Body(..., embed=True),
    password: str = Body(..., embed=True),
    session_id: Optional[str] = Body(default=None, embed=True),
):
    """注册新用户：校验 -> 入库 -> 直接签发 token（自动登录）。

    跟登录同样：尝试回填 session_id，失败则新建一个 session 给新用户并绑定。
    """
    if not _USERNAME_RE.match(username):
        raise HTTPException(
            status_code=400,
            detail="用户名需为 3-32 位字母、数字、下划线或中文",
        )
    if not password or len(password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 位")
    if crud.get_user_by_username(username):
        raise HTTPException(status_code=409, detail="该用户名已被注册")
    user_id = crud.create_user(username, hash_password(password))
    user = crud.get_user_by_id(user_id)
    token = create_token(user_id, user["token_version"])
    refresh_token = create_refresh_token(user_id, user["token_version"])

    # 与登录一致：尝试回填 session_id，失败则新建一个 session 绑定到新用户
    final_session_id = None
    if session_id:
        try:
            ok = manager.reassign_user_to_session(session_id, crud.to_user_id_str(user_id))
            if ok:
                final_session_id = session_id
        except Exception as e:
            logger.warning("register session 回填失败 session_id=%s: %s", session_id, e)
    if not final_session_id:
        final_session_id = manager.assign_new_session_to_user(user_id)

    return {
        "token": token,
        "refresh_token": refresh_token,
        "user": {"id": user_id, "username": username},
        "session_id": final_session_id,
    }


@router.post("/login")
def login(
    username: str = Body(..., embed=True),
    password: str = Body(..., embed=True),
    session_id: Optional[str] = Body(default=None, embed=True),
):
    """登录：校验密码 -> 签发 token -> 回填游客 session（带时间窗校验）。

    回填逻辑：
      - 传了 session_id 且该 session「非空 + 未绑定 + 30 分钟内活跃」 → 回填；
      - 其余情况（无传 / session 不存在 / 是空 session / 时间窗过期）→ 新建一个 session
        并立即绑定到当前登录用户，返回给前端。

    返回 session_id 是关键：前端必须用它覆盖 localStorage.sessionId，
    否则后续用户上传文件会落到 user_id=NULL 的旧 session 上，
    下次冷启动 clear_guest_data 会被误删。
    """
    user = crud.get_user_by_username(username)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_token(user["id"], user["token_version"])
    refresh_token = create_refresh_token(user["id"], user["token_version"])

    # 登录即绑定当前游客 session（方案 A）
    final_session_id = None
    if session_id:
        try:
            ok = manager.reassign_user_to_session(session_id, crud.to_user_id_str(user["id"]))
            if ok:
                final_session_id = session_id
        except Exception as e:
            # 回填失败不影响登录成功（兜底：用户仍拿到 token），仅记录日志
            logger.warning("session 回填失败 session_id=%s: %s", session_id, e)

    # 没成功回填：永远给登录用户一个「全新」的 session。
    # 这样用户重新登录后默认是一个干净的对话窗口，不会自动弹出旧会话。
    # 历史会话只能通过 /history 页面显式点击恢复，避免「退出登录再次登录后
    # 旧会话自动出现」的体验问题。
    if not final_session_id:
        final_session_id = manager.assign_new_session_to_user(user["id"])

    return {
        "token": token,
        "refresh_token": refresh_token,
        "user": {"id": user["id"], "username": user["username"]},
        "session_id": final_session_id,
    }


@router.post("/refresh")
def refresh(refresh_token: str = Body(..., embed=True)):
    """用 refresh token 换发新的 access token（不校验旧 access 是否过期）。"""
    new_access = refresh_access_token(refresh_token)
    return {"token": new_access}


@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    """返回当前登录用户信息（供前端恢复会话）。"""
    return {"id": user["id"], "username": user["username"]}


@router.post("/change-password")
def change_password(
    old_password: str = Body(..., embed=True),
    new_password: str = Body(..., embed=True),
    user: dict = Depends(get_current_user),
):
    """修改密码：校验旧密码 -> 更新哈希 -> token_version +1（旧 token 失效）。"""
    if not verify_password(old_password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="原密码错误")
    if not new_password or len(new_password) < 6:
        raise HTTPException(status_code=400, detail="新密码至少 6 位")
    crud.update_password(user["id"], hash_password(new_password))
    new_version = crud.revoke_user_tokens(user["id"])
    return {"success": True, "token_version": new_version}


@router.post("/logout")
def logout(user: dict = Depends(get_current_user)):
    """退出登录：token_version +1，使当前及所有旧 token 立即失效。"""
    new_version = crud.revoke_user_tokens(user["id"])
    return {"success": True, "token_version": new_version}
