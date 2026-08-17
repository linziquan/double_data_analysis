"""
CRUD 封装：所有对 SQLite 的读写集中于此，业务代码（SessionManager）不直接写 SQL。

约定：
- sessions.state_json：会话可序列化的全部轻量状态（JSON 字符串）。
- datasets：记录 DataFrame 落盘 pickle 的持久化路径 + 元信息。
- analysis_packages：AnalysisPackage 完整 JSON。

线程安全：调用方（SessionManager）已用 RLock 串行化；本层每次操作取连接执行，
连接本身为线程局部，避免 sqlite 跨线程错误。
"""
import json
import logging
import os
import sqlite3
import time
from typing import Optional, Dict, Any, List

from .connection import get_connection

logger = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    """数据集数量超过用户配额时抛出，由路由转换为 403 QUOTA_EXCEEDED。"""


def _sanitize_nonfinite(obj: Any) -> Any:
    """递归把 NaN / +Inf / -Inf 替换为 None，避免 JSON 序列化报
    'Out of range float values are not JSON compliant'（往返数据库时死锁）。

    - 标量 float: math.isfinite 判断；非有限 → None
    - dict / list: 递归
    - tuple / set: 列表化以保持 JSON 可序列化
    - str / int / bool / None: 原样
    """
    import math
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nonfinite(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_nonfinite(v) for v in obj]
    if isinstance(obj, set):
        return [_sanitize_nonfinite(v) for v in obj]
    return obj


def _to_json(obj: Any) -> str:
    return json.dumps(_sanitize_nonfinite(obj), ensure_ascii=False, default=str)


def _from_json(text: str) -> Any:
    return _sanitize_nonfinite(json.loads(text))


def to_user_id_str(user_id: Any) -> Optional[str]:
    """将 user_id 统一转为 TEXT 列所需的可空字符串。

    - None / 空串 -> None（游客）
    - int / str -> str(int) 或 str，与 users.id 对齐，杜绝 int→TEXT 隐式转换污染。
    """
    if user_id is None:
        return None
    s = str(user_id).strip()
    if s == "" or s.lower() == "none":
        return None
    return s


# ===================== sessions =====================

def _ensure_sessions_user_column(conn) -> None:
    """幂等为 sessions 表补充 user_id 列（历史库可能缺失）。"""
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)")]
        if "user_id" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT")
            conn.commit()
    except Exception as e:
        logger.warning("ensure sessions.user_id 失败: %s", e)


def save_session_state(session_id: str, state: Dict[str, Any], created_at: float,
                       last_access: float, user_id: Optional[str] = None) -> None:
    """写入/更新会话状态（UPSERT）。user_id 透传，便于按用户归集历史会话。"""
    conn = get_connection()
    _ensure_sessions_user_column(conn)
    uid = to_user_id_str(user_id)
    conn.execute(
        """
        INSERT INTO sessions (session_id, state_json, created_at, last_access, user_id)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            state_json = excluded.state_json,
            last_access = excluded.last_access,
            user_id = COALESCE(excluded.user_id, sessions.user_id)
        """,
        (session_id, _to_json(state), created_at, last_access, uid),
    )
    conn.commit()


def list_sessions_by_user(user_id: Any) -> List[Dict[str, Any]]:
    """返回该用户的全部「有效」会话摘要（按最后访问倒序）。

    摘要字段：session_id, title, last_page, dataset_count, package_count,
              created_at, last_access。title 取会话 custom_title 或一个数据集名兜底。

    过滤策略（修复"登出再登录多出空记录"）：
      只列出 state_json 中 dataset_packages 或 analysis_packages 至少 1 个的会话。
      即"用户实际在该会话中投入了工作"的会话。
      过滤掉：
        - 游客态 DataProvider 自动创建的空 session（即便被回填/兜底创建成用户 session 也无数据）
        - 登录路径 assign_new_session_to_user 创建的空 session（这些是登录后兜底
          给用户一个有效 sessionId；用户必须真正上传文件才能成为有效会话）
      只有真正上传过文件或跑过分析的 session 才进入历史。
    """
    uid = to_user_id_str(user_id)
    if uid is None:
        return []
    conn = get_connection()
    _ensure_sessions_user_column(conn)
    rows = conn.execute(
        """
        SELECT session_id, state_json, created_at, last_access, user_id
        FROM sessions WHERE user_id = ? ORDER BY last_access DESC
        """,
        (uid,),
    ).fetchall()
    result = []
    for r in rows:
        try:
            state = _from_json(r["state_json"])
        except Exception:
            state = {}
        # 过滤空 session。判定"用户实际使用过"的口径：任一即可
        #   1) state_json.dataset_packages 至少 1 个桶非空（用户在 UI 上传过数据集、跑过分析）
        #   2) state_json.analysis_packages 非空（兜底；老格式）
        #   3) state_json.chat_history 有 ≥1 条消息（用户问过 LLM 即视作"会话记录"）
        #   4) datasets 表里有 ≥1 行（防御：state_json 与 datasets 表不同步的极端场景）
        #   5) analysis_packages 表里有 ≥1 行（防御：同上）
        # 旧的「必须 dataset_packages 或 analysis_packages 非空」过于苛刻，
        # 导致「只问问题、还没出分析包」的会话被丢出历史列表（用户预期是"历史记录 = 过去聊过的会话"）。
        try:
            sid = r["session_id"]
            dp = state.get("dataset_packages") or {}
            ap = state.get("analysis_packages") or {}
            ch = state.get("chat_history") or []
            state_has_work = (
                (isinstance(dp, dict) and any(isinstance(v, dict) and v for v in dp.values()))
                or (isinstance(ap, dict) and len(ap) > 0)
                or (isinstance(ch, list) and len(ch) > 0)
            )
            if not state_has_work:
                # 兜底再查一次 db 表
                nonempty = conn.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM datasets          WHERE session_id = ?) AS d_count,
                      (SELECT COUNT(*) FROM analysis_packages WHERE session_id = ?) AS p_count
                    """,
                    (sid, sid),
                ).fetchone()
                if not (bool(nonempty) and (int(nonempty["d_count"] or 0) > 0
                                              or int(nonempty["p_count"] or 0) > 0)):
                    continue
        except Exception:
            continue
        # 取一个数据集名作为会话标题兜底
        title = state.get("custom_title") or ""
        # 数据集 / 分析包数量：优先以数据库表为准（更准确），
        # state_json.dataset_packages 仅在内存层缓存，可能与持久化层不一致。
        # 例如新上传的数据集还没回流到 state_json 时，UI 上能看到数据集但历史列表显示 0。
        ds_count = 0
        try:
            ds_row = conn.execute(
                "SELECT COUNT(*) AS n FROM datasets WHERE session_id = ?",
                (sid,),
            ).fetchone()
            if ds_row and ds_row["n"]:
                ds_count = int(ds_row["n"])
        except Exception:
            pass
        if ds_count == 0:
            try:
                ds_count = len(state.get("dataset_packages", {}) or {})
            except Exception:
                ds_count = 0
        first_ds_name = ""
        try:
            for bucket in (state.get("dataset_packages", {}) or {}).values():
                if isinstance(bucket, dict):
                    for pkg in bucket.values():
                        if isinstance(pkg, dict) and pkg.get("file_name"):
                            first_ds_name = pkg.get("file_name")
                            break
                if first_ds_name:
                    break
        except Exception:
            first_ds_name = ""
        # 兜底：datasets 表里有但 state_json 没缓存时，从数据集表里取文件名做标题
        if not first_ds_name and ds_count > 0:
            try:
                row = conn.execute(
                    "SELECT meta_json FROM datasets WHERE session_id = ? LIMIT 1",
                    (sid,),
                ).fetchone()
                if row and row["meta_json"]:
                    meta = _from_json(row["meta_json"])
                    first_ds_name = (
                        (meta or {}).get("file_name")
                        or (meta or {}).get("original_filename")
                        or ""
                    )
            except Exception:
                pass
        if not title:
            title = first_ds_name or "未命名会话"
        pkg_count = 0
        try:
            ap_row = conn.execute(
                "SELECT COUNT(*) AS n FROM analysis_packages WHERE session_id = ?",
                (sid,),
            ).fetchone()
            if ap_row and ap_row["n"]:
                pkg_count = int(ap_row["n"])
        except Exception:
            pass
        if pkg_count == 0:
            try:
                pkg_count = len(state.get("analysis_packages", {}) or {})
            except Exception:
                pkg_count = 0
        result.append({
            "session_id": r["session_id"],
            "title": title,
            "last_page": state.get("last_page") or "upload",
            "dataset_count": ds_count,
            "package_count": pkg_count,
            "chat_count": (
                len(state.get("chat_history") or [])
                if isinstance(state.get("chat_history"), list)
                else 0
            ),
            "created_at": r["created_at"],
            "last_access": r["last_access"],
        })
    return result


def get_latest_session_for_user(user_id: Any) -> Optional[str]:
    """取该用户最近访问的会话 id（不限是否有数据/分析），

    用于登录兜底时恢复用户上一次的登录会话（含纯对话会话）。
    游客会话（user_id IS NULL）不会被返回，从而「不保留游客历史」自然成立。
    """
    uid = to_user_id_str(user_id)
    if uid is None:
        return None
    conn = get_connection()
    _ensure_sessions_user_column(conn)
    row = conn.execute(
        "SELECT session_id FROM sessions WHERE user_id = ? ORDER BY last_access DESC LIMIT 1",
        (uid,),
    ).fetchone()
    return row["session_id"] if row else None


def load_session_state(session_id: str) -> Optional[Dict[str, Any]]:
    """读取会话状态；不存在返回 None。"""
    conn = get_connection()
    row = conn.execute(
        "SELECT state_json, created_at, last_access FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    state = _from_json(row["state_json"])
    state["_created_at"] = row["created_at"]
    state["_last_access"] = row["last_access"]
    return state


def touch_session(session_id: str, last_access: float) -> None:
    """仅更新会话最后访问时间。"""
    conn = get_connection()
    conn.execute(
        "UPDATE sessions SET last_access = ? WHERE session_id = ?",
        (last_access, session_id),
    )
    conn.commit()


def delete_session(session_id: str) -> None:
    """删除会话及其全部数据集、分析包（级联清理）。"""
    conn = get_connection()
    conn.execute("DELETE FROM analysis_packages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM datasets WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()


def delete_session_by_user(user_id: Any, session_id: str) -> tuple[bool, list[str]]:
    """带归属校验的会话删除：仅当会话归属该 user_id 才删除。

    返回 (ok, pkl_paths)：ok 表示是否真的删除了该用户会话；
    pkl_paths 是该会话下需要物理删除的数据集 pkl 文件路径（路由层负责 IO 删除）。
    """
    uid = to_user_id_str(user_id)
    if uid is None:
        return (False, [])
    conn = get_connection()
    row = conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ? AND user_id = ?",
        (session_id, uid),
    ).fetchone()
    if row is None:
        return (False, [])
    # 收集 pkl 路径（用于路由层物理清理）
    pkl_paths = [
        r["original_path"]
        for r in conn.execute(
            "SELECT original_path FROM datasets WHERE session_id = ?", (session_id,)
        ).fetchall()
        if r["original_path"]
    ]
    conn.execute("DELETE FROM analysis_packages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM datasets WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    return (True, pkl_paths)


def clear_all_data() -> None:
    """清空全部上传数据（sessions / datasets / analysis_packages 三表全清）。

    用途：后端冷启动时释放所有历史数据，恢复到空白状态。幂等、不依赖内存状态。
    """
    conn = get_connection()
    conn.execute("DELETE FROM analysis_packages")
    conn.execute("DELETE FROM datasets")
    conn.execute("DELETE FROM sessions")
    conn.commit()


def list_expired_sessions(timeout: float, now: float) -> List[str]:
    """返回最后访问距 now 超过 timeout 秒的会话 ID 列表。"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT session_id FROM sessions WHERE ? - last_access > ?",
        (now, timeout),
    ).fetchall()
    return [r["session_id"] for r in rows]


# ===================== datasets =====================

def save_dataset(session_id: str, dataset_id: str, meta: Dict[str, Any],
                 original_path: str, is_active: bool, created_at: float,
                 user_id: Optional[str] = None) -> None:
    """写入/更新数据集记录。

    user_id：可空（游客为 None）。登录用户写入前做配额校验（dataset_limit），
    超额抛 QuotaExceededError，由路由转换为 403 QUOTA_EXCEEDED。
    """
    uid = to_user_id_str(user_id)
    # 登录用户配额校验：统计该用户已有数据集数量（仅首次落盘的数据集计入，避免多 sheet 重复计）
    if uid is not None:
        limit = get_user_dataset_limit(int(uid))
        count = count_user_datasets(int(uid))
        if count >= limit:
            raise QuotaExceededError(
                f"数据集数量已达上限（{count}/{limit}），请删除部分历史数据集后再上传"
            )
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO datasets (dataset_id, session_id, meta_json, original_path, is_active, created_at, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id) DO UPDATE SET
            meta_json = excluded.meta_json,
            original_path = excluded.original_path,
            is_active = excluded.is_active,
            user_id = excluded.user_id
        """,
        (dataset_id, session_id, _to_json(meta), original_path,
         1 if is_active else 0, created_at, uid),
    )
    conn.commit()


def _all_dataset_metas(session_id: str) -> List[Dict[str, Any]]:
    """读取某会话全部数据集的元信息（含 dataset_id / 落盘路径）。仅供 SessionManager 内部 hydrate 使用。"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT dataset_id, meta_json, original_path, is_active FROM datasets WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    result = []
    for r in rows:
        meta = _from_json(r["meta_json"])
        meta["dataset_id"] = r["dataset_id"]
        meta["original_path"] = r["original_path"]
        meta["is_active"] = bool(r["is_active"])
        result.append(meta)
    return result


def load_dataset_meta(dataset_id: str) -> Optional[Dict[str, Any]]:
    """读取数据集元信息 + 落盘路径。"""
    conn = get_connection()
    row = conn.execute(
        "SELECT meta_json, original_path, is_active FROM datasets WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchone()
    if row is None:
        return None
    meta = _from_json(row["meta_json"])
    meta["original_path"] = row["original_path"]
    meta["is_active"] = bool(row["is_active"])
    return meta


def set_dataset_active(session_id: str, dataset_id: str) -> None:
    """将某数据集设为该会话 active，其余置为非 active。"""
    conn = get_connection()
    conn.execute("UPDATE datasets SET is_active = 0 WHERE session_id = ?", (session_id,))
    conn.execute(
        "UPDATE datasets SET is_active = 1 WHERE dataset_id = ? AND session_id = ?",
        (dataset_id, session_id),
    )
    conn.commit()


def delete_dataset(session_id: str, dataset_id: str) -> None:
    conn = get_connection()
    conn.execute(
        "DELETE FROM analysis_packages WHERE dataset_id = ? AND session_id = ?",
        (dataset_id, session_id),
    )
    conn.execute(
        "DELETE FROM datasets WHERE dataset_id = ? AND session_id = ?",
        (dataset_id, session_id),
    )
    conn.commit()


# ===================== analysis_packages =====================

def save_package(package_id: str, session_id: str, dataset_id: str,
                 payload: Dict[str, Any], saved_at: Optional[str],
                 created_at: float, user_id: Optional[str] = None) -> None:
    """写入/更新分析包。user_id 透传，便于按用户归集/隔离。"""
    uid = to_user_id_str(user_id)
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO analysis_packages (package_id, session_id, dataset_id, payload_json, saved_at, created_at, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(package_id) DO UPDATE SET
            payload_json = excluded.payload_json,
            saved_at = excluded.saved_at,
            dataset_id = excluded.dataset_id,
            user_id = excluded.user_id
        """,
        (package_id, session_id, dataset_id, _to_json(payload),
         saved_at, created_at, uid),
    )
    conn.commit()


def load_package(package_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        "SELECT payload_json FROM analysis_packages WHERE package_id = ?",
        (package_id,),
    ).fetchone()
    return _from_json(row["payload_json"]) if row else None


def load_packages_by_session(session_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT package_id, dataset_id, payload_json, saved_at FROM analysis_packages WHERE session_id = ?",
        (session_id,),
    ).fetchall()
    result = []
    for r in rows:
        pkg = _from_json(r["payload_json"])
        pkg["_dataset_id"] = r["dataset_id"]
        pkg["_saved_at"] = r["saved_at"]
        result.append(pkg)
    return result


def load_packages_by_dataset(session_id: str, dataset_id: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT payload_json FROM analysis_packages WHERE session_id = ? AND dataset_id = ?",
        (session_id, dataset_id),
    ).fetchall()
    return [_from_json(r["payload_json"]) for r in rows]


def delete_package(package_id: str) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM analysis_packages WHERE package_id = ?", (package_id,))
    conn.commit()


# ===================== users =====================

def create_user(username: str, password_hash: str) -> int:
    """创建用户，返回新用户 id。"""
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO users (username, password_hash, token_version, created_at) VALUES (?, ?, 0, ?)",
        (username, password_hash, time.time()),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        "SELECT id, username, password_hash, token_version, storage_used, dataset_limit FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute(
        "SELECT id, username, password_hash, token_version, storage_used, dataset_limit FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


def get_user_token_version(user_id: int) -> Optional[int]:
    """返回用户当前 token_version；用户不存在返回 None。"""
    conn = get_connection()
    row = conn.execute("SELECT token_version FROM users WHERE id = ?", (user_id,)).fetchone()
    return int(row["token_version"]) if row else None


def update_password(user_id: int, password_hash: str) -> None:
    conn = get_connection()
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
    conn.commit()


def revoke_user_tokens(user_id: int) -> int:
    """token_version +1，使该用户所有旧 token 失效。返回新的 version。"""
    conn = get_connection()
    conn.execute("UPDATE users SET token_version = token_version + 1 WHERE id = ?", (user_id,))
    conn.commit()
    return get_user_token_version(user_id)


def get_user_dataset_limit(user_id: int) -> int:
    conn = get_connection()
    row = conn.execute("SELECT dataset_limit FROM users WHERE id = ?", (user_id,)).fetchone()
    return int(row["dataset_limit"]) if row else 50


def count_user_datasets(user_id: int) -> int:
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM datasets WHERE user_id = ?", (str(user_id),)
    ).fetchone()
    return int(row["c"]) if row else 0


# ===================== 数据归属回溯 =====================

def session_has_real_work(session_id: str) -> bool:
    """判定 session 是否「有真实工作」（至少 1 个数据集 或 1 个分析包）。

    用于登录回填路径上拒绝"空游客 session"——避免 0 数据 0 分析包的空记录进入历史列表。
    """
    conn = get_connection()
    row = conn.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM datasets          WHERE session_id = ?) AS d_count,
          (SELECT COUNT(*) FROM analysis_packages WHERE session_id = ?) AS p_count
        """,
        (session_id, session_id),
    ).fetchone()
    return bool(row) and (row["d_count"] > 0 or row["p_count"] > 0)


def reassign_session_to_user(session_id: str, user_id: str, window_seconds: int = 1800) -> bool:
    """将游客 session 回填到登录用户（方案 A）。

    安全约束：
      - 存在
      - user_id IS NULL（尚未绑定）
      - last_access 在 window_seconds（默认 30 分钟）内活跃
      - 该 session 至少有 1 个数据集或 1 个分析包（防止「空游客 session」被回填成
        0 数据 0 分析包的噪声历史记录：用户进入应用时 DataProvider 会自动创建一个
        空游客 session，若此时立刻登录会触发回填，导致历史列表里堆满空记录）
    才执行回填，杜绝越权绑定他人 session。
    返回是否成功回填。
    """
    uid = to_user_id_str(user_id)
    if uid is None:
        return False
    conn = get_connection()
    cutoff = time.time() - window_seconds
    # 先单独判「非空」：至少 1 个 dataset 或 1 个 analysis_package
    nonempty_check = conn.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM datasets        WHERE session_id = ? AND user_id IS NULL) AS d_count,
          (SELECT COUNT(*) FROM analysis_packages WHERE session_id = ? AND user_id IS NULL) AS p_count
        """,
        (session_id, session_id),
    ).fetchone()
    if not nonempty_check or (nonempty_check["d_count"] == 0 and nonempty_check["p_count"] == 0):
        # 空 session 不绑定：直接返回 False；该 session 在游客端继续存在（user_id 仍为 NULL），
        # 不会被 list_sessions_by_user 列出，不污染历史。游客后续如要继续工作，可重新登录后
        # 调 /api/session/new 拿新 sessionId 替换。
        conn.commit()
        return False
    cur = conn.execute(
        """
        UPDATE sessions
        SET user_id = ?
        WHERE session_id = ? AND user_id IS NULL AND last_access > ?
        """,
        (uid, session_id, cutoff),
    )
    affected = cur.rowcount
    if affected > 0:
        # 同步回填该 session 下的数据集与分析包（事务）
        conn.execute(
            "UPDATE datasets SET user_id = ? WHERE session_id = ? AND user_id IS NULL",
            (uid, session_id),
        )
        conn.execute(
            "UPDATE analysis_packages SET user_id = ? WHERE session_id = ? AND user_id IS NULL",
            (uid, session_id),
        )
    conn.commit()
    # 会话数量上限：每个用户最多保留 MAX_SESSIONS_PER_USER 条，超出则删除最旧的
    if affected > 0:
        prune_user_sessions(uid)
    return affected > 0


# 每个用户保留的历史会话上限（超出删除最旧）
MAX_SESSIONS_PER_USER = 10


def prune_user_sessions(user_id: Any, max_count: int = MAX_SESSIONS_PER_USER) -> int:
    """删除该用户最旧的会话，使其总数不超过 max_count。返回被删除的会话数。

    策略（修复"登出再登录多出空记录"）：
      1. 优先删除「空 session」（state_json.dataset_packages / analysis_packages 都为空），
         即那些只是登录路径兜底创建但用户从未真正上传/分析过的会话。
      2. 实在不够删（全是有效 session）才按 last_access 升序删最旧的（保留用户真正的工作）。

    这避免「用户登录创建空 session → 占满 10 条限额 → 真正上传文件时把老有效 session 误删」。
    """
    uid = to_user_id_str(user_id)
    if uid is None:
        return 0
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT session_id, state_json, last_access FROM sessions
        WHERE user_id = ? ORDER BY last_access ASC
        """,
        (uid,),
    ).fetchall()
    if len(rows) <= max_count:
        return 0
    excess_n = len(rows) - max_count
    deleted = 0

    def _is_empty(sid: str) -> bool:
        # state_json 视角：dataset_packages / analysis_packages 均为空 → 视为空 session
        try:
            srow = conn.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?", (sid,)
            ).fetchone()
            if not srow:
                return True
            state = _from_json(srow["state_json"]) or {}
            dp = state.get("dataset_packages") or {}
            ap = state.get("analysis_packages") or {}
            for bucket in (dp if isinstance(dp, dict) else {}).values():
                if isinstance(bucket, dict) and len(bucket) > 0:
                    return False
            if isinstance(ap, dict) and len(ap) > 0:
                return False
            return True
        except Exception:
            return False

    # 第一轮：按 last_access 升序，优先删空 session
    for r in rows:
        if deleted >= excess_n:
            break
        sid = r["session_id"]
        if not _is_empty(sid):
            continue
        try:
            delete_session(sid)
            deleted += 1
        except Exception as e:
            logger.warning("prune_user_sessions 第一轮删除 %s 失败: %s", sid, e)
    # 第二轮：若仍超额（说明全是有效 session），按 last_access 升序删最旧的（保留用户最新工作）
    if deleted < excess_n:
        for r in rows:
            if deleted >= excess_n:
                break
            sid = r["session_id"]
            # 不重复删除（已在第一轮删了的还在 transactions 内，此刻就不存在了）
            srow = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (sid,)
            ).fetchone()
            if not srow:
                continue
            try:
                delete_session(sid)
                deleted += 1
            except Exception as e:
                logger.warning("prune_user_sessions 第二轮删除 %s 失败: %s", sid, e)
    return deleted


# ===================== 历史查询（A+B 合并：按数据集分组 + 挂分析包） =====================

def list_datasets_by_user(user_id: int) -> List[Dict[str, Any]]:
    """返回该用户全部数据集（按创建时间倒序），含 meta 解析与数据集 ID。"""
    uid = str(user_id)
    conn = get_connection()
    rows = conn.execute(
        "SELECT dataset_id, meta_json, original_path, created_at FROM datasets WHERE user_id = ? ORDER BY created_at DESC",
        (uid,),
    ).fetchall()
    result = []
    for r in rows:
        meta = _from_json(r["meta_json"])
        meta["dataset_id"] = r["dataset_id"]
        meta["original_path"] = r["original_path"]
        meta["created_at"] = r["created_at"]
        result.append(meta)
    return result


def list_packages_by_user(user_id: int, dataset_id: str) -> List[Dict[str, Any]]:
    """返回该用户某数据集下的全部分析包。"""
    uid = str(user_id)
    conn = get_connection()
    rows = conn.execute(
        "SELECT package_id, payload_json, saved_at, created_at FROM analysis_packages WHERE user_id = ? AND dataset_id = ? ORDER BY created_at DESC",
        (uid, dataset_id),
    ).fetchall()
    result = []
    for r in rows:
        pkg = _from_json(r["payload_json"])
        pkg["package_id"] = r["package_id"]
        pkg["saved_at"] = r["saved_at"]
        pkg["created_at"] = r["created_at"]
        result.append(pkg)
    return result


def delete_dataset_by_user(user_id: int, dataset_id: str) -> List[str]:
    """删除某用户的数据集及其下全部分析包，返回被删除的 pkl 路径清单（供调用方物理删文件）。

    已确认 owner 一致才删（防越权删除）；返回 original_path 列表。
    """
    uid = str(user_id)
    conn = get_connection()
    rows = conn.execute(
        "SELECT original_path FROM datasets WHERE user_id = ? AND dataset_id = ?",
        (uid, dataset_id),
    ).fetchall()
    paths = [r["original_path"] for r in rows]
    if not paths:
        return []
    conn.execute(
        "DELETE FROM analysis_packages WHERE user_id = ? AND dataset_id = ?",
        (uid, dataset_id),
    )
    conn.execute(
        "DELETE FROM datasets WHERE user_id = ? AND dataset_id = ?",
        (uid, dataset_id),
    )
    conn.commit()
    return paths


def delete_package_by_user(user_id: int, package_id: str) -> bool:
    """删除某用户的单个分析包（级联清理其收藏记录）。"""
    uid = str(user_id)
    conn = get_connection()
    cur = conn.execute(
        "DELETE FROM analysis_packages WHERE user_id = ? AND package_id = ?",
        (uid, package_id),
    )
    conn.execute(
        "DELETE FROM favorites WHERE user_id = ? AND package_id = ?",
        (uid, package_id),
    )
    conn.commit()
    return cur.rowcount > 0


# ===================== 冷启动清理（只清游客） =====================

def clear_guest_data() -> int:
    """冷启动清理：仅删除游客（user_id IS NULL）数据，保留登录用户及其落盘 pkl。

    实现（Bug 3）：先 SELECT 收集游客数据集的 original_path，再删行，最后删 pkl 文件。
    任一文件 IO 失败仅 logging.warning，不抛错，保证登录用户数据零误删。
    返回删除的游客数据集行数。
    """
    conn = get_connection()
    # 1) 先收路径（删行前，避免路径信息丢失）
    guest_rows = conn.execute(
        "SELECT original_path FROM datasets WHERE user_id IS NULL OR user_id = ''"
    ).fetchall()
    guest_paths = [r["original_path"] for r in guest_rows if r["original_path"]]

    # 2) 删行（analysis_packages -> datasets -> sessions）
    conn.execute("DELETE FROM analysis_packages WHERE user_id IS NULL OR user_id = ''")
    conn.execute("DELETE FROM datasets WHERE user_id IS NULL OR user_id = ''")
    conn.execute("DELETE FROM sessions WHERE user_id IS NULL OR user_id = ''")
    conn.commit()
    deleted_count = len(guest_paths)

    # 3) 物理删 pkl（最后，且失败不阻断）
    for p in guest_paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception as e:
            logger.warning("删除游客 pkl 失败（已忽略）: %s -> %s", p, e)
    return deleted_count


# ===================== 收藏 / 分组（P2） =====================

def _ensure_favorites_table(conn: sqlite3.Connection) -> None:
    """幂等创建 favorites 表与索引（兼容升级前旧库）。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS favorites (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       TEXT NOT NULL,
            package_id    TEXT NOT NULL,
            is_starred    INTEGER NOT NULL DEFAULT 0,
            display_name  TEXT,
            group_name    TEXT NOT NULL DEFAULT '默认分组',
            sort_order    INTEGER NOT NULL DEFAULT 0,
            created_at    REAL NOT NULL,
            UNIQUE(user_id, package_id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_favorites_user ON favorites(user_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_favorites_user_pkg ON favorites(user_id, package_id)"
    )


def toggle_favorite(user_id: int, package_id: str, starred: bool) -> Dict[str, Any]:
    """收藏 / 取消收藏某分析包。starred=True 置 is_starred=1，否则置 0（保留分组/重命名）。

    返回最新收藏记录（含 is_starred / group_name / display_name）。
    """
    uid = str(user_id)
    conn = get_connection()
    _ensure_favorites_table(conn)
    now = time.time()
    existing = conn.execute(
        "SELECT id, group_name, display_name, sort_order FROM favorites WHERE user_id = ? AND package_id = ?",
        (uid, package_id),
    ).fetchone()

    if existing is None:
        conn.execute(
            """
            INSERT INTO favorites (user_id, package_id, is_starred, display_name, group_name, sort_order, created_at)
            VALUES (?, ?, ?, NULL, '默认分组', 0, ?)
            """,
            (uid, package_id, 1 if starred else 0, now),
        )
        group_name = "默认分组"
        display_name = None
        sort_order = 0
    else:
        conn.execute(
            "UPDATE favorites SET is_starred = ? WHERE user_id = ? AND package_id = ?",
            (1 if starred else 0, uid, package_id),
        )
        group_name = existing["group_name"]
        display_name = existing["display_name"]
        sort_order = existing["sort_order"]
    conn.commit()
    return {
        "package_id": package_id,
        "is_starred": starred,
        "group_name": group_name,
        "display_name": display_name,
        "sort_order": sort_order,
    }


def set_favorite_meta(user_id: int, package_id: str, display_name: Optional[str] = None,
                      group_name: Optional[str] = None, sort_order: Optional[int] = None) -> Dict[str, Any]:
    """设置收藏项的显示名 / 分组名 / 排序。仅对已有收藏记录生效，否则 404。"""
    uid = str(user_id)
    conn = get_connection()
    _ensure_favorites_table(conn)
    existing = conn.execute(
        "SELECT id FROM favorites WHERE user_id = ? AND package_id = ?",
        (uid, package_id),
    ).fetchone()
    if existing is None:
        raise KeyError("收藏记录不存在")
    fields = []
    params: List[Any] = []
    if display_name is not None:
        fields.append("display_name = ?")
        params.append(display_name or None)
    if group_name is not None:
        fields.append("group_name = ?")
        params.append(group_name)
    if sort_order is not None:
        fields.append("sort_order = ?")
        params.append(sort_order)
    if fields:
        params.append(uid)
        params.append(package_id)
        conn.execute(
            f"UPDATE favorites SET {', '.join(fields)} WHERE user_id = ? AND package_id = ?",
            params,
        )
        conn.commit()
    row = conn.execute(
        "SELECT package_id, is_starred, display_name, group_name, sort_order FROM favorites WHERE user_id = ? AND package_id = ?",
        (uid, package_id),
    ).fetchone()
    return dict(row)


def list_favorites_by_user(user_id: int) -> List[Dict[str, Any]]:
    """列出某用户的全部收藏（含分析包快照：标题/类型/时间），按分组 + 排序返回。

    返回结构：[{ group_name, items: [{ package_id, is_starred, display_name, sort_order,
              title, package_type, created_at, saved_at }] }]
    """
    uid = str(user_id)
    conn = get_connection()
    _ensure_favorites_table(conn)
    rows = conn.execute(
        """
        SELECT f.package_id, f.is_starred, f.display_name, f.group_name, f.sort_order,
               p.payload_json, p.saved_at, p.created_at AS pkg_created_at
        FROM favorites f
        LEFT JOIN analysis_packages p ON p.package_id = f.package_id
        WHERE f.user_id = ? AND f.is_starred = 1
        ORDER BY f.group_name, f.sort_order, f.created_at
        """,
        (uid,),
    ).fetchall()

    groups: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for r in rows:
        payload = _from_json(r["payload_json"]) if r["payload_json"] else {}
        title = payload.get("title") or payload.get("custom_title") or "未命名分析"
        pkg_type = payload.get("package_type") or payload.get("type") or "unknown"
        item = {
            "package_id": r["package_id"],
            "is_starred": bool(r["is_starred"]),
            "display_name": r["display_name"],
            "sort_order": r["sort_order"],
            "title": title,
            "package_type": pkg_type,
            "created_at": r["pkg_created_at"],
            "saved_at": r["saved_at"],
        }
        g = r["group_name"]
        if g not in groups:
            groups[g] = []
            order.append(g)
        groups[g].append(item)
    return [{"group_name": g, "items": groups[g]} for g in order]


def get_favorite_state(user_id: int, package_id: str) -> Optional[Dict[str, Any]]:
    """查询某分析包的收藏状态（用于历史页图标高亮）。不存在返回 None。"""
    uid = str(user_id)
    conn = get_connection()
    _ensure_favorites_table(conn)
    row = conn.execute(
        "SELECT package_id, is_starred, display_name, group_name, sort_order FROM favorites WHERE user_id = ? AND package_id = ?",
        (uid, package_id),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


# ===================== 分享链接（P2） =====================

def _ensure_shares_table(conn: sqlite3.Connection) -> None:
    """幂等创建 shares 表与索引（兼容升级前旧库）。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shares (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            share_id   TEXT UNIQUE NOT NULL,
            package_id TEXT NOT NULL,
            user_id    TEXT NOT NULL,
            created_at REAL NOT NULL,
            expire_at  REAL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shares_share_id ON shares(share_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shares_user ON shares(user_id)")


def create_share(user_id: int, package_id: str, expire_at: Optional[float] = None) -> Dict[str, Any]:
    """为某分析包创建公开只读分享，返回 { share_id, package_id, expire_at }。

    同一用户同一分析包重复分享时复用已存在的 share_id（幂等）。
    """
    import uuid
    uid = str(user_id)
    conn = get_connection()
    _ensure_shares_table(conn)
    now = time.time()
    existing = conn.execute(
        "SELECT share_id, expire_at FROM shares WHERE user_id = ? AND package_id = ?",
        (uid, package_id),
    ).fetchone()
    if existing is not None:
        # 幂等复用：但允许用户重新设置过期时间（覆盖旧值）
        conn.execute(
            "UPDATE shares SET expire_at = ? WHERE share_id = ?",
            (expire_at, existing["share_id"]),
        )
        conn.commit()
        return {
            "share_id": existing["share_id"],
            "package_id": package_id,
            "expire_at": expire_at,
        }
    share_id = uuid.uuid4().hex[:12]
    conn.execute(
        """
        INSERT INTO shares (share_id, package_id, user_id, created_at, expire_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (share_id, package_id, uid, now, expire_at),
    )
    conn.commit()
    return {"share_id": share_id, "package_id": package_id, "expire_at": expire_at}


def get_share(share_id: str) -> Optional[Dict[str, Any]]:
    """按 share_id 取分享记录（含 package payload）。过期或不存在返回 None。"""
    conn = get_connection()
    _ensure_shares_table(conn)
    row = conn.execute(
        "SELECT package_id, user_id, created_at, expire_at FROM shares WHERE share_id = ?",
        (share_id,),
    ).fetchone()
    if row is None:
        return None
    if row["expire_at"] is not None and time.time() > row["expire_at"]:
        return None
    pkg = conn.execute(
        "SELECT payload_json FROM analysis_packages WHERE package_id = ?",
        (row["package_id"],),
    ).fetchone()
    payload = _from_json(pkg["payload_json"]) if pkg and pkg["payload_json"] else None
    return {
        "share_id": share_id,
        "package_id": row["package_id"],
        "user_id": row["user_id"],
        "created_at": row["created_at"],
        "expire_at": row["expire_at"],
        "payload": payload,
    }


def list_shares_by_user(user_id: int) -> List[Dict[str, Any]]:
    """列出某用户的全部分享（不含 payload，仅元信息）。"""
    uid = str(user_id)
    conn = get_connection()
    _ensure_shares_table(conn)
    rows = conn.execute(
        "SELECT share_id, package_id, created_at, expire_at FROM shares WHERE user_id = ? ORDER BY created_at DESC",
        (uid,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_share_by_user(user_id: int, share_id: str) -> bool:
    """删除某用户的某个分享（仅本人可删）。"""
    uid = str(user_id)
    conn = get_connection()
    _ensure_shares_table(conn)
    cur = conn.execute(
        "DELETE FROM shares WHERE user_id = ? AND share_id = ?",
        (uid, share_id),
    )
    conn.commit()
    return cur.rowcount > 0

