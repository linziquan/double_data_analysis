"""
会话管理器 - 替代 Streamlit session_state
使用 UUID 作为 session_id，DataFrame 存储在内存中
"""

import os
import uuid
import time
import json
import tempfile
import logging
import dataclasses
import pandas as pd
from typing import Dict, Optional, List, Any
from threading import RLock, Thread
from config import QUOTA_BYTES
from backend.db import crud
logger = logging.getLogger(__name__)


def _safe_payload(pkg):
    """把分析包统一转成可序列化 dict；无法转换的脏数据返回 None（跳过）。"""
    if pkg is None:
        return None
    if isinstance(pkg, dict):
        return pkg
    # 治本兜底：历史脏 state 里残留 JSON 字符串（如旧版把整个包 repr 成字符串落库），
    # 这里尝试还原为 dict，避免刷屏、并救回旧会话的分析列表。
    if isinstance(pkg, str):
        try:
            obj = json.loads(pkg)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        logger.debug("跳过无法解析的分析包字符串(历史脏数据): %s",
                     (pkg[:40] + "..." if len(pkg) > 40 else pkg))
        return None
    if dataclasses.is_dataclass(pkg):
        return dataclasses.asdict(pkg)
    try:
        return dict(pkg)
    except Exception:
        # 只打印包的身份标识，避免 repr() 整个包（含图表数据点）刷屏
        try:
            ident = getattr(pkg, "id", None) or getattr(pkg, "analysis_type", None) or type(pkg).__name__
        except Exception:
            ident = type(pkg).__name__
        logger.warning("跳过无法序列化的分析包: %s", ident)
        return None


@dataclasses.dataclass
class Dataset:
    """单个数据集（每张上传的报表对应一个）"""
    dataset_id: str
    file_name: str
    file_size_bytes: int
    df: Optional[pd.DataFrame] = None          # 非 active 数据集可为 None（pickle 保留，按需 reload）
    df_original: Optional[pd.DataFrame] = None  # 仅落盘失败时的兜底内存副本；正常为 None
    original_path: Optional[str] = None        # 原始数据落盘路径(pickle)，释放内存时保留
    rows: int = 0
    columns: List[str] = dataclasses.field(default_factory=list)
    column_info: List[dict] = dataclasses.field(default_factory=list)
    preview: List[dict] = dataclasses.field(default_factory=list)
    uploaded_at: float = 0.0
    # 多表合并标记（合并宽表专用）
    is_merged: bool = False                    # 是否为合并生成的宽表
    sources: List[str] = dataclasses.field(default_factory=list)   # 来源 dataset_id 列表
    merge_keys: List[str] = dataclasses.field(default_factory=list) # 实际使用的关联键列名
    # 标记：该数据集已被「AI 智能清洗」流水线做过「合表 + 映射 + LLM 清洗」。
    # 分析阶段 _process_one 检测到则跳过重复映射，省一次调用。
    cleaned_mapped: bool = False


def _parse_missing_rate(row) -> float:
    """从 column_info 行解析缺失率（兼容百分比字符串 '12.3%' 与纯数字）。"""
    try:
        v = row.get("缺失率")
        if v is None:
            return 0.0
        if isinstance(v, str):
            return float(v.replace("%", "").strip()) / 100.0
        return float(v)
    except Exception:
        return 0.0


class SessionData:
    """单个会话的数据（支持多数据集）"""
    def __init__(self):
        # ===== 多数据集存储 =====
        self.datasets: Dict[str, Dataset] = {}        # key=dataset_id
        self.active_dataset_id: Optional[str] = None  # 当前分析对象
        self.uploaded_bytes: int = 0                  # 累计已上传字节（=Σ file_size_bytes）
        self.dataset_packages: Dict[str, Dict[str, Any]] = {}  # dataset_id→{pkg_id:pkg}
        # 向后兼容：df / df_original / original_path 改为委托到 active 数据集的属性（见下方 property）
        self.df_undo_stack: List[pd.DataFrame] = []  # 撤销栈（最多保存 20 步）
        self.cleaning_history: List[Dict] = []
        self.analysis_history: List[Dict] = []
        # 任务1：AI 会话多轮聊天历史（chat/analyze 追问回路统一维护）
        # 每条形如 {"role": "user"|"assistant"|"tool", "content": str, "ts": float}
        self.chat_history: List[Dict[str, Any]] = []
        # 任务1：用户偏好档案卡（防偏好锁死：只记"用户最近的真实业务意图"，不记风格取向）
        # 形如 {"last_business_question": "哪个品卖得不好"}，AI 决策时查此卡避免重复反问，
        # 但用户最新指令优先（见任务3 系统提示词）。写入即覆盖。
        self.user_preferences: Dict[str, Any] = {}
        self.saved_charts: List[Dict[str, Any]] = []  # 用户从分析页保存的图表 [{"title":..., "option":..., "saved_at":...}, ...]
        self.analysis_packages: Dict[str, Any] = {}     # 临时分析结果（key=pkg_id, value=AnalysisPackage）
        self.saved_packages: List[Dict[str, Any]] = []   # 用户保存的分析包
        self.api_key: str = ""
        self.ai_provider: str = ""           # AI 服务商（deepseek/openai/custom 等）
        self.custom_model: str = ""          # 自定义模型名
        self.custom_base_url: str = ""       # 自定义 API Base URL
        self.custom_title: str = ""          # 用户手动编辑的仪表盘标题
        self.holds_slot: bool = False        # 是否已占用"数据插槽"（限流=持有数据的会话，上限 max_sessions）
        self.reserved_at: float = 0.0         # 占用插槽的时间戳（用于预约超时释放）
        self.user_id: Optional[str] = None    # 登录用户归属（方案 A）：None=游客；回填后存 user_id 字符串
        self.last_page: str = "upload"         # 会话"上次最后访问页面"，用于历史恢复智能跳转
        self.created_at: float = time.time()
        self.last_access: float = time.time()
        # ===== Chat 智能体扩展 =====
        self.data_profile: Dict[str, Any] = {}   # 上传后 data_recon 侦察结果
        self.messages: List[Dict[str, Any]] = []  # 智能体对话历史（LLM messages 数组）

    # ===== df / df_original / original_path 委托到 active 数据集（向后兼容下游 ~30 处 get_data 调用）=====
    def _active_dataset(self) -> Optional["Dataset"]:
        if self.active_dataset_id is not None:
            return self.datasets.get(self.active_dataset_id)
        return None

    @property
    def df(self) -> Optional[pd.DataFrame]:
        ds = self._active_dataset()
        return ds.df if ds else None

    @df.setter
    def df(self, value: Optional[pd.DataFrame]):
        ds = self._active_dataset()
        if ds is not None:
            ds.df = value

    @property
    def df_original(self) -> Optional[pd.DataFrame]:
        ds = self._active_dataset()
        return ds.df_original if ds else None

    @df_original.setter
    def df_original(self, value: Optional[pd.DataFrame]):
        ds = self._active_dataset()
        if ds is not None:
            ds.df_original = value

    @property
    def original_path(self) -> Optional[str]:
        ds = self._active_dataset()
        return ds.original_path if ds else None

    @original_path.setter
    def original_path(self, value: Optional[str]):
        ds = self._active_dataset()
        if ds is not None:
            ds.original_path = value


class SessionManager:
    """会话管理器，线程安全的内存存储"""
    
    def __init__(self, max_sessions: int = 5, session_timeout: int = 604800):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = RLock()
        self._max_sessions = max_sessions
        self._session_timeout = session_timeout
        # 原始数据落盘目录（持久化到 data/，重启后可按路径 reload，不再丢失）
        # 注：data/ 已加入根 .gitignore，防大文件进仓库（遵循 30MB 上传限制纪律）。
        self._original_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "originals"
        )
        os.makedirs(self._original_dir, exist_ok=True)
        # P2（内存画像结论五）：后台定时清理线程，主动回收过期会话（弥补惰性清理短板）。
        # 间隔跟随 timeout：max(60, timeout//12) → 默认 3600//12 = 300s。
        # 最坏滞后 = 间隔，过期会话最多比 timeout 多赖 300s，内存卫生足够及时且不浪费 0.1 CPU。
        self._cleanup_interval = max(60, session_timeout // 12)
        # 排队队列与已晋升映射（限流相关，均在锁内访问）
        self._queue: List[Dict[str, Any]] = []          # FIFO: {ticket_id, session_id, created_at}
        self._promoted: Dict[str, str] = {}             # ticket_id -> session_id（已晋升等待上传）
        self._QUEUE_TTL = 300                           # 排队票据最长等待（秒），超时丢弃
        self._RESERVE_TTL = 120                         # 预约插槽但未上传的最长保留（秒），超时释放
        self._slot_idle_timeout = 600                   # 未保存会话空闲超时（秒）：释放内存+删落盘+让槽
        self._saved_idle_timeout = 3600                 # 已保存会话更长空闲阈值（秒）：仅释放内存+让槽、保留落盘
        self._start_background_cleanup()
    
    # ===== 持久化（SQLite）：内存缓存 + 落库，保留全部公共方法签名 =====
    def _normalize_package_value(self, pkg):
        """规范化单个分析包 value 为可序列化 dict；无法转换返回 None（丢弃脏数据）。

        覆盖：AnalysisPackage 对象→asdict、dict→原样、JSON 字符串→解析还原、
        其他→None。保证写进 SQLite state 的永远是干净 dict，杜绝重启后字符串复现。
        """
        if pkg is None:
            return None
        if isinstance(pkg, dict):
            return pkg
        if isinstance(pkg, str):
            try:
                obj = json.loads(pkg)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
            return None
        if dataclasses.is_dataclass(pkg):
            return dataclasses.asdict(pkg)
        try:
            d = dict(pkg)
            return d
        except Exception:
            return None

    def _serialize_session(self, session: "SessionData") -> Dict[str, Any]:
        """将 SessionData 中可序列化的轻量状态抽出（不含 DataFrame 本体）。

        落库前把 dataset_packages/analysis_packages/saved_packages 三个包字段的
        value 规范化为干净 dict，避免 AnalysisPackage 对象或 JSON 字符串被整体塞进
        state 表、重启后还原成字符串导致刷屏与列表为空（治本）。
        """
        norm_dataset_packages = {}
        for did, bucket in (session.dataset_packages or {}).items():
            if isinstance(bucket, dict):
                norm_bucket = {pid: self._normalize_package_value(pkg) for pid, pkg in bucket.items()}
                norm_bucket = {pid: p for pid, p in norm_bucket.items() if p is not None}
                if norm_bucket:
                    norm_dataset_packages[did] = norm_bucket
            else:
                norm_bucket = self._normalize_package_value(bucket)
                if norm_bucket is not None:
                    norm_dataset_packages[did] = {"_single": norm_bucket}
        norm_analysis = {
            pid: p for pid, p in (
                (pid, self._normalize_package_value(pkg))
                for pid, pkg in (session.analysis_packages or {}).items()
            ) if p is not None
        }
        norm_saved = [p for p in (session.saved_packages or []) if isinstance(p, dict)]
        return {
            "active_dataset_id": session.active_dataset_id,
            "uploaded_bytes": session.uploaded_bytes,
            "dataset_packages": norm_dataset_packages,
            "analysis_packages": norm_analysis,
            "saved_packages": norm_saved,
            "api_key": session.api_key,
            "ai_provider": session.ai_provider,
            "custom_model": session.custom_model,
            "custom_base_url": session.custom_base_url,
            "custom_title": session.custom_title,
            "holds_slot": session.holds_slot,
            "reserved_at": session.reserved_at,
            "user_id": session.user_id,
            "last_page": session.last_page,
            "cleaning_history": session.cleaning_history,
            "analysis_history": session.analysis_history,
            "chat_history": session.chat_history,
            "user_preferences": session.user_preferences,
            "saved_charts": session.saved_charts,
            "df_undo_stack": session.df_undo_stack,
            "created_at": session.created_at,
            "last_access": session.last_access,
        }

    def _hydrate_session(self, session: "SessionData", state: Dict[str, Any]) -> None:
        """用从库读取的 state 填充 SessionData（不含 DataFrame）。"""
        session.active_dataset_id = state.get("active_dataset_id")
        session.uploaded_bytes = state.get("uploaded_bytes", 0)
        session.dataset_packages = state.get("dataset_packages", {}) or {}
        session.analysis_packages = state.get("analysis_packages", {}) or {}
        session.saved_packages = state.get("saved_packages", []) or []
        session.api_key = state.get("api_key", "")
        session.ai_provider = state.get("ai_provider", "")
        session.custom_model = state.get("custom_model", "")
        session.custom_base_url = state.get("custom_base_url", "")
        session.custom_title = state.get("custom_title", "")
        session.holds_slot = state.get("holds_slot", False)
        session.reserved_at = state.get("reserved_at", 0.0)
        session.user_id = state.get("user_id")
        session.last_page = state.get("last_page") or "upload"
        session.cleaning_history = state.get("cleaning_history", []) or []
        session.analysis_history = state.get("analysis_history", []) or []
        session.chat_history = state.get("chat_history", []) or []
        session.user_preferences = state.get("user_preferences", {}) or {}
        session.saved_charts = state.get("saved_charts", []) or []
        session.df_undo_stack = state.get("df_undo_stack", []) or []
        if state.get("_created_at") is not None:
            session.created_at = state["_created_at"]
        if state.get("_last_access") is not None:
            session.last_access = state["_last_access"]

    def _persist_session(self, session_id: str) -> None:
        """将内存会话状态与数据集/分析包落库（在锁内调用）。"""
        session = self._sessions.get(session_id)
        if session is None:
            return
        # 1) 数据集元信息 + 落盘路径
        for ds in session.datasets.values():
            crud.save_dataset(
                session_id, ds.dataset_id,
                {
                    "file_name": ds.file_name,
                    "file_size_bytes": ds.file_size_bytes,
                    "rows": ds.rows,
                    "columns": ds.columns,
                    "column_info": ds.column_info,
                    "preview": ds.preview,
                    "is_merged": ds.is_merged,
                    "sources": ds.sources,
                    "merge_keys": ds.merge_keys,
                    "uploaded_at": ds.uploaded_at,
                },
                ds.original_path or "",
                1 if ds.dataset_id == session.active_dataset_id else 0,
                ds.uploaded_at or time.time(),
                user_id=session.user_id,
            )
        # 2) 分析包（B5 修复：用真实 dataset_id，保证历史按文件分组非空）
        # dataset_packages 是 {dataset_id: {pkg_id: pkg}}，第一层 key 即真实 dataset_id
        for did, bucket in (session.dataset_packages or {}).items():
            if isinstance(bucket, dict):
                for pid, pkg in bucket.items():
                    payload = _safe_payload(pkg)
                    if payload is not None:
                        crud.save_package(pid, session_id, did, payload, None, time.time(),
                                     user_id=session.user_id)
        # analysis_packages / saved_packages 是扁平结构，从 pkg 内部 dataset_id 取（缺失则回退空）
        for pid, pkg in (session.analysis_packages or {}).items():
            payload = _safe_payload(pkg)
            if payload is not None:
                _did = payload.get("dataset_id") or ""
                crud.save_package(pid, session_id, _did, payload, None, time.time(),
                                 user_id=session.user_id)
        for pkg in session.saved_packages:
            pid = pkg.get("id")
            if pid:
                _did = pkg.get("dataset_id") or ""
                crud.save_package(pid, session_id, _did, pkg, pkg.get("saved_at"), time.time(),
                                 user_id=session.user_id)
        # 3) 会话轻量状态
        crud.save_session_state(
            session_id, self._serialize_session(session),
            session.created_at, session.last_access,
            user_id=session.user_id,
        )

    def _load_session_from_db(self, session_id: str) -> Optional["SessionData"]:
        """会话不在内存时，尝试从 SQLite 重建（不含 DataFrame，需按需 reload）。"""
        state = crud.load_session_state(session_id)
        if state is None:
            return None
        session = SessionData()
        self._hydrate_session(session, state)
        # 重建数据集对象（含 original_path，DataFrame 按需 reload）
        for ds_meta in crud._all_dataset_metas(session_id):
            ds = Dataset(
                dataset_id=ds_meta["dataset_id"],
                file_name=ds_meta.get("file_name", ""),
                file_size_bytes=ds_meta.get("file_size_bytes", 0),
                original_path=ds_meta.get("original_path", ""),
                rows=ds_meta.get("rows", 0),
                columns=ds_meta.get("columns", []),
                column_info=ds_meta.get("column_info", []),
                preview=ds_meta.get("preview", []),
                is_merged=ds_meta.get("is_merged", False),
                sources=ds_meta.get("sources", []),
                merge_keys=ds_meta.get("merge_keys", []),
                uploaded_at=ds_meta.get("uploaded_at", 0.0),
            )
            session.datasets[ds.dataset_id] = ds
        # 兜底：脏会话 datasets 字段为 JSON null 时重建后为 None，遍历会 500（fd22be77 根因）
        session.datasets = _safe_dict(session.datasets)
        self._sessions[session_id] = session
        return session

    # ===== 限流：数据插槽预约 / 排队 / 晋升 =====
    def _slot_count(self) -> int:
        """当前已占用数据插槽的会话数（锁内调用）"""
        return sum(1 for s in self._sessions.values() if s.holds_slot)

    def acquire_for_upload(self, session_id: str) -> Dict[str, Any]:
        """预约数据插槽；满员则把该会话入队。

        返回 {'granted': bool, 'session_id'?, 'ticket_id'?, 'position'?}
        - granted=True：已预约（或已有数据），可立即上传，附 session_id
        - granted=False：已满员，附 ticket_id 与当前排队位次 position（1 起）
        """
        with self._lock:
            self._cleanup_sync()
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session
            # 已占插槽或已有数据 → 直接放行（幂等，避免重复预约）
            if session.holds_slot or session.active_dataset_id is not None:
                return {"granted": True, "session_id": session_id,
                        "used_bytes": session.uploaded_bytes, "quota_bytes": QUOTA_BYTES}
            # 有空位 → 预约该会话
            if self._slot_count() < self._max_sessions:
                session.holds_slot = True
                session.reserved_at = time.time()
                session.last_access = time.time()
                return {"granted": True, "session_id": session_id,
                        "used_bytes": session.uploaded_bytes, "quota_bytes": QUOTA_BYTES}
            # 满员 → 入队，返回位次
            ticket_id = str(uuid.uuid4())
            self._queue.append({
                "ticket_id": ticket_id,
                "session_id": session_id,
                "created_at": time.time(),
            })
            return {"granted": False, "ticket_id": ticket_id, "position": len(self._queue)}

    def queue_status(self, ticket_id: str) -> Dict[str, Any]:
        """查询排队状态。

        返回 {'status': 'ready'|'queued'|'expired', 'session_id'?, 'position'?}
        - ready：已晋升，附可上传的 session_id
        - queued：仍在等待，附当前位次 position（1 起）
        - expired：票据不存在或已失效（会话丢失）
        """
        with self._lock:
            for i, item in enumerate(self._queue):
                if item["ticket_id"] == ticket_id:
                    return {"status": "queued", "position": i + 1}
            if ticket_id in self._promoted:
                sid = self._promoted[ticket_id]
                sess = self._sessions.get(sid)
                if sess is not None and sess.holds_slot:
                    return {"status": "ready", "session_id": sid, "position": 0}
                # 晋升后会话丢失 → 视为过期
                self._promoted.pop(ticket_id, None)
            return {"status": "expired"}

    def cancel_queue(self, ticket_id: str) -> None:
        """从等待队列移除票据（尽力而为；已晋升项无法撤回上传，仅移除映射）。"""
        with self._lock:
            self._queue = [it for it in self._queue if it["ticket_id"] != ticket_id]
            self._promoted.pop(ticket_id, None)

    def _promote_head(self) -> None:
        """晋升队首到就绪（锁内调用）。循环 drained 直至无队首或无空位。"""
        while self._queue and self._slot_count() < self._max_sessions:
            item = self._queue.pop(0)
            sid = item["session_id"]
            session = self._sessions.get(sid)
            if session is None:
                # 队首会话已不存在 → 新建会话承接票据，避免丢票
                sid = str(uuid.uuid4())
                session = SessionData()
                self._sessions[sid] = session
            session.holds_slot = True
            session.reserved_at = time.time()
            session.last_access = time.time()
            self._promoted[item["ticket_id"]] = sid

    def reserve_session(self, session_id: str) -> None:
        """为已有会话占用一个数据插槽（上传兜底路径用，正常前端已预占必有空位）。"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session
            if not session.holds_slot and session.df is None:
                session.holds_slot = True
                session.reserved_at = time.time()
                session.last_access = time.time()

    def _release_slot_inner(self, session_id: str) -> bool:
        """锁内复用：释放某会话的数据插槽（丢弃 df 与落盘原文件，但保留会话对象）。

        不含晋升，由调用方在持锁状态下统一调 _promote_head，避免重复加锁。
        返回是否真的释放了一个插槽。
        """
        session = self._sessions.get(session_id)
        if session is None or not session.holds_slot:
            return False
        self._remove_original_file(session_id)
        session.df = None
        session.holds_slot = False
        return True

    def release_slot(self, session_id: str) -> bool:
        """释放某会话的数据插槽（保留会话对象以便重新上传，但丢弃 DataFrame 与原文件以释放内存）。

        释放后自动晋升队首。返回是否真的释放了一个插槽。
        这是「自动入队」的现实触发点之一：手动释放（API/按钮）与服务端
        空闲超时（_slot_idle_timeout）都会经此路径腾出插槽。
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.holds_slot:
                return False
            self._release_slot_inner(session_id)
            session.last_access = time.time()
            self._promote_head()
            return True

    def create_session(self) -> str:
        """创建新会话，返回 session_id（不再淘汰最老会话，限流改为按数据插槽）。"""
        session_id = str(uuid.uuid4())
        with self._lock:
            # 清理过期会话
            self._cleanup_sync()
            self._sessions[session_id] = SessionData()
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """获取会话数据；内存未命中时尝试从 SQLite 重建（含数据集与落盘路径）。"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = self._load_session_from_db(session_id)
            if session:
                session.last_access = time.time()
                crud.touch_session(session_id, session.last_access)
            return session

    def set_current_page(self, session_id: str, page: str) -> None:
        """记录会话当前页面，供历史恢复时智能跳转。"""
        if not page:
            return
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = self._load_session_from_db(session_id)
            if session is None:
                return
            session.last_page = page
            session.last_access = time.time()
            # 轻量落库：仅更新 state（含 last_page）与 last_access
            crud.save_session_state(
                session_id, self._serialize_session(session),
                session.created_at, session.last_access,
                user_id=session.user_id,
            )

    def reassign_user_to_session(self, session_id: str, user_id: str) -> bool:
        """登录回填：把游客 session 归属到登录用户（方案 A）。

        安全策略：
          - 必须有真实工作（至少 1 个 datasets 或 1 个 analysis_packages），
            否则视为空游客 session，拒绝绑定（防止 history 列表堆满 0 数据 0 分析包的空记录）。
          - 通过后，更新内存 session.user_id + 落库，再做 DB 层时间窗回填。
        返回是否成功回填。
        """
        uid = crud.to_user_id_str(user_id)
        if uid is None:
            return False
        # 先做非空判定（DB 层），空 session 完全不碰 → 返回 False 让 auth 路由走新建兜底
        try:
            if not crud.session_has_real_work(session_id):
                return False
        except Exception as e:
            import logging as _logging
            _logging.getLogger("session").warning(f"非空判定失败（按拒绝处理）: {e}")
            return False
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.user_id = uid
                self._persist_session(session_id)  # 内存态落库，携带 user_id
        # 再对 DB 做带时间窗的安全回填（覆盖重启后内存丢失的场景）
        try:
            return crud.reassign_session_to_user(session_id, uid)
        except Exception as e:
            import logging as _logging
            _logging.getLogger("session").warning(f"reassign_session_to_user DB 回填失败: {e}")
            # 内存态已更新，视为成功（至少本次会话归属正确）
            return True

    def assign_new_session_to_user(self, user_id) -> str:
        """登录路径：新建一个会话并绑定到登录用户，避免空游客 session 污染历史。

        - 新建 session（uuid）→ 立即落库并 user_id=user_id 绑定
        - 返回新 session_id 给前端，前端用其覆盖 localStorage.sessionId
        用于：登录请求带的 session_id 是空 session 时（crud.reassign_session_to_user 会返回 False），
        仍要给登录用户一个有效的、归属正确的 session 以便后续上传文件落库。

        注意：游客态的对话不保留（用户需求：仅保留登录用户的会话历史）。
        登录用户自己的历史会话靠 user_id 归属 + crud.list_sessions_by_user 恢复，不在此复制。
        """
        uid = crud.to_user_id_str(user_id)
        if uid is None:
            return ""
        new_sid = self.create_session()
        with self._lock:
            session = self._sessions.get(new_sid)
            if session is not None:
                session.user_id = uid
                self._persist_session(new_sid)  # 落库，携带 user_id
        # 库里同步回填（虽然内存已设，但这次在 DB 上也绑定一次以保一致；同时触发 prune_user_sessions）
        try:
            crud.reassign_session_to_user(new_sid, uid)
        except Exception as e:
            import logging as _logging
            _logging.getLogger("session").warning(f"assign_new_session_to_user 绑定失败: {e}")
        return new_sid

    def add_dataset(self, session_id: str, df: pd.DataFrame, *, file_name: str,
                    file_size_bytes: int, rows: int, columns: List[str],
                    column_info: List[dict], preview: List[dict],
                    dataset_id: Optional[str] = None, set_active: bool = True,
                    account_quota: bool = True, user_id: Optional[str] = None) -> str:
        """新增一个数据集（不覆盖旧表）。返回 dataset_id。

        原始数据落盘(pickle)以释放内存；非 active 数据集仅保留 pickle、释放内存 df（防 OOM）。
        落盘失败(磁盘满/权限)时兜底保留内存副本，保证功能不丢。
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session
            # 去除重复列名，避免后续 df[col] 返回 DataFrame 而非 Series
            if df.columns.duplicated().any():
                dup_cols = df.columns[df.columns.duplicated()].unique().tolist()
                import logging as _logging; _logging.getLogger("session").warning(f"removing duplicate columns: {dup_cols}")
                df = df.loc[:, ~df.columns.duplicated()]
            did = dataset_id or str(uuid.uuid4())
            if did in session.datasets:
                did = str(uuid.uuid4())  # 重名则重新生成，避免覆盖
            path = os.path.join(self._original_dir, f"{session_id}_{did}.pkl")
            try:
                df.to_pickle(path)
                original_path = path
                df_original = None  # 正常路径不保留内存副本
            except Exception as e:  # 兜底：落盘失败则保留内存副本
                import logging as _logging
                _logging.getLogger("session").warning(f"原始数据落盘失败，回退内存保留: {e}")
                original_path = None
                df_original = df.copy()
            ds = Dataset(
                dataset_id=did, file_name=file_name, file_size_bytes=file_size_bytes,
                df=df.copy(), df_original=df_original, original_path=original_path,
                rows=rows, columns=list(columns), column_info=list(column_info),
                preview=list(preview), uploaded_at=time.time(),
            )
            session.datasets[did] = ds
            # account_quota=False 时只落库不累计额度（多 sheet 文件在首个 sheet 已计一次）
            session.uploaded_bytes += file_size_bytes if account_quota else 0
            if set_active:
                session.active_dataset_id = did
                # 修复一：驱逐其余非 active 数据集的内存 df（pickle 保留）
                for other in session.datasets.values():
                    if other.dataset_id != did and other.df is not None:
                        other.df = None
                # 闭环：新数据集成为 active，同步其（可能为空）产物
                session.analysis_packages = dict(session.dataset_packages.get(did, {}))
                # ★ 新数据集激活时，清理基于旧 df 的 saved_packages / saved_charts，
                #   避免「上传新文件后仪表盘仍然显示旧测试数据的 KPI/charts」问题。
                #   SELECT_DATASET 切回旧数据集不清空，仍可看到该数据集已保存的分析结果。
                session.saved_packages = []
                session.saved_charts = []
                session.analysis_packages = {}
            session.last_access = time.time()
            self._persist_session(session_id)
            return did

    def _compute_meta(self, df: "pd.DataFrame"):
        """从 df 计算 (rows, columns, column_info, preview)，供 add_merged_dataset / update_dataset_df 复用。"""
        import numpy as np
        from src.data_loader import get_column_info, get_data_info
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        preview = df.head(100).replace({np.nan: None}).to_dict(orient="records")
        column_info_df = get_column_info(df)
        columns_list = []
        for _, row in column_info_df.iterrows():
            columns_list.append({
                "name": str(row.get("列名", row.get("column", ""))),
                "dtype": str(row.get("数据类型", row.get("dtype", ""))),
                "missing": int(row.get("缺失值", row.get("missing", 0)) or 0),
                "missing_rate": _parse_missing_rate(row),
                "unique": int(row.get("唯一值数", row.get("unique", 0)) or 0),
                "sample": str(row.get("示例值", row.get("sample", ""))),
            })
        data_info = get_data_info(df)
        rows = int(data_info.get("行数", len(df)))
        return rows, list(df.columns), columns_list, preview

    def update_dataset_df(self, session_id: str, dataset_id: str, df: "pd.DataFrame"):
        """按数据集 ID 写回清洗后的 df，并重算元信息（rows/columns/column_info/preview）。
        锁内仅替换 ds.df 副本，不动 original_path / df_original（保留「恢复原始数据」能力）。
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            ds = session.datasets.get(dataset_id)
            if ds is None:
                return
            rows, columns, column_info, preview = self._compute_meta(df)
            ds.df = df.copy()
            ds.rows = rows
            ds.columns = columns
            ds.column_info = column_info
            ds.preview = preview
            session.last_access = time.time()

    def add_merged_dataset(self, session_id: str, df: pd.DataFrame,
                           sources: List[str], keys: List[str],
                           file_name: str = "合并宽表") -> str:
        """合并宽表入库：构造与上传一致的元信息，在锁内注册新数据集
        （set_active=False，不抢占当前视图），并补写 is_merged/sources/merge_keys，
        返回新 dataset_id。

        合并阶段在 process-datasets 流水线中调用，宽表一旦注册即可被下游
        列名映射与规则分析流水线正常识别。
        """
        import logging as _logging
        import numpy as np
        from src.data_loader import get_column_info, get_data_info

        # 先去除重复列名（合并可能引入同名非键列），再构造元信息
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        preview = df.head(100).replace({np.nan: None}).to_dict(orient="records")
        column_info_df = get_column_info(df)
        columns_list = []
        for _, row in column_info_df.iterrows():
            columns_list.append({
                "name": str(row.get("列名", row.get("column", ""))),
                "dtype": str(row.get("数据类型", row.get("dtype", ""))),
                "missing": int(row.get("缺失值", row.get("missing", 0)) or 0),
                "missing_rate": _parse_missing_rate(row),
                "unique": int(row.get("唯一值数", row.get("unique", 0)) or 0),
                "sample": str(row.get("示例值", row.get("sample", ""))),
            })
        data_info = get_data_info(df)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session
            did = str(uuid.uuid4())
            while did in session.datasets:
                did = str(uuid.uuid4())
            path = os.path.join(self._original_dir, f"{session_id}_{did}.pkl")
            try:
                df.to_pickle(path)
                original_path = path
                df_original = None
            except Exception as e:
                _logging.getLogger("session").warning(f"合并宽表落盘失败，回退内存保留: {e}")
                original_path = None
                df_original = df.copy()
            ds = Dataset(
                dataset_id=did, file_name=file_name,
                file_size_bytes=int(df.memory_usage(deep=True).sum()),
                df=df.copy(), df_original=df_original, original_path=original_path,
                rows=int(data_info.get("行数", len(df))),
                columns=list(df.columns), column_info=list(columns_list),
                preview=list(preview), uploaded_at=time.time(),
                is_merged=True, sources=list(sources), merge_keys=list(keys),
            )
            session.datasets[did] = ds
            # 不抢占当前 active 视图、不驱逐、不计额度
            session.last_access = time.time()
            self._persist_session(session_id)
            return did

    def set_data(self, session_id: str, df: pd.DataFrame):
        """向后兼容：等价于新增一个默认 dataset（老调用方用）"""
        self.add_dataset(
            session_id, df,
            file_name="data",
            file_size_bytes=int(df.memory_usage(deep=True).sum()),
            rows=int(df.shape[0]), columns=list(df.columns),
            column_info=[], preview=[],
        )
    
    def get_data(self, session_id: str) -> Optional[pd.DataFrame]:
        """获取当前 DataFrame"""
        session = self.get_session(session_id)
        return session.df if session else None
    
    def get_original_data(self, session_id: str) -> Optional[pd.DataFrame]:
        """获取原始 DataFrame（从磁盘读取；不存在/损坏返回 None）。

        会话存在但文件已被重启清除 -> 返回 None，由调用方提示"原始数据已释放"。
        """
        session = self.get_session(session_id)
        if session is None:
            return None
        ds = session._active_dataset()
        if ds is None:
            return None
        # 兜底：落盘失败时的内存副本
        if ds.df_original is not None:
            return ds.df_original
        if ds.original_path and os.path.exists(ds.original_path):
            try:
                return pd.read_pickle(ds.original_path)
            except Exception:
                return None
        return None

    def _remove_original_file(self, session_id: str):
        """删除会话对应所有数据集的原始数据落盘文件（须在锁内调用；忽略异常）"""
        session = self._sessions.get(session_id)
        if session:
            self._clear_session_datasets(session)

    def _clear_session_datasets(self, session: SessionData):
        """清空会话全部数据集（删除落盘 + 释放内存 + 归零额度）；须在锁内"""
        for ds in list(session.datasets.values()):
            if ds.original_path and os.path.exists(ds.original_path):
                try:
                    os.remove(ds.original_path)
                except OSError:
                    pass
        session.datasets.clear()
        session.active_dataset_id = None
        session.uploaded_bytes = 0
        session.df = None
        session.df_original = None
        session.original_path = None

    # ===== 多数据集新方法 =====
    def get_dataset_df(self, session_id: str, dataset_id: str) -> Optional[pd.DataFrame]:
        """获取指定数据集的 df（缺失则从 pickle reload 回内存）

        兼容早期版本：DB 里的 original_path 可能指向失效的旧路径基准，
        这里按多个候选顺序尝试 {session_id}_{did}.pkl / {did}.pkl 命名规则。
        """
        session = self.get_session(session_id)
        if session is None:
            return None
        ds = session.datasets.get(dataset_id)
        if ds is None:
            return None
        if ds.df is not None:
            return ds.df
        candidate_paths = []
        if ds.original_path:
            candidate_paths.append(ds.original_path)
        candidate_paths.append(os.path.join(self._original_dir, f"{session_id}_{dataset_id}.pkl"))
        candidate_paths.append(os.path.join(self._original_dir, f"{dataset_id}.pkl"))
        for cand in candidate_paths:
            if cand and os.path.exists(cand):
                try:
                    ds.df = pd.read_pickle(cand)
                    if ds.original_path != cand:
                        ds.original_path = cand
                    return ds.df
                except Exception:
                    continue
        if ds.df_original is not None:
            ds.df = ds.df_original
            return ds.df
        return None

    def select_dataset(self, session_id: str, dataset_id: str) -> bool:
        """切换当前分析对象（active）；按需 reload + 驱逐其余非 active 内存 df（修复一）"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or dataset_id not in session.datasets:
                return False
            session.active_dataset_id = dataset_id
            # 闭环：切换后把当前数据集的分析产物同步进 analysis_packages（看板/报告统一读取入口）
            session.analysis_packages = dict(session.dataset_packages.get(dataset_id, {}))
            ds = session.datasets[dataset_id]
            # 按需 reload 内存 df
            if ds.df is None:
                if ds.original_path and os.path.exists(ds.original_path):
                    try:
                        ds.df = pd.read_pickle(ds.original_path)
                    except Exception:
                        if ds.df_original is not None:
                            ds.df = ds.df_original
                elif ds.df_original is not None:
                    ds.df = ds.df_original
            # 驱逐其余非 active 数据集的内存 df（pickle 保留）
            for other in session.datasets.values():
                if other.dataset_id != dataset_id and other.df is not None:
                    other.df = None
            session.last_access = time.time()
            self._persist_session(session_id)
            return True

    def get_datasets(self, session_id: str) -> List[Dict[str, Any]]:
        """返回全部数据集的元信息列表（供前端"已上传报表"列表 / 刷新拉回）"""
        session = self.get_session(session_id)
        if session is None:
            return []
        result = []
        for ds in _safe_dict(session.datasets).values():
            try:
                result.append({
                    "dataset_id": ds.dataset_id,
                    "file_name": ds.file_name,
                    "file_size_bytes": ds.file_size_bytes,
                    "rows": ds.rows,
                    "columns": ds.columns,
                    "column_info": ds.column_info,
                    "preview": ds.preview,
                    "uploaded_at": ds.uploaded_at,
                    "is_active": ds.dataset_id == session.active_dataset_id,
                    "is_merged": ds.is_merged,
                    "sources": ds.sources,
                    "merge_keys": ds.merge_keys,
                })
            except Exception as e:
                # 单条坏数据不应拖垮整个列表接口：记录并跳过该条
                bad_id = getattr(ds, "dataset_id", "<unknown>")
                logger.error("[get_datasets] 跳过损坏数据集 dataset_id=%s: %s",
                             bad_id, e)
                continue
        # 按上传时间倒序（最新在前）
        result.sort(key=lambda x: x.get("uploaded_at", 0), reverse=True)
        return result

    def remove_dataset(self, session_id: str, dataset_id: str) -> bool:
        """删除指定数据集（删落盘 + 减额度 + 回退 active）"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or dataset_id not in session.datasets:
                return False
            ds = session.datasets.pop(dataset_id)
            # 删落盘
            if ds.original_path and os.path.exists(ds.original_path):
                try:
                    os.remove(ds.original_path)
                except OSError:
                    pass
            # 减额度
            session.uploaded_bytes = max(0, session.uploaded_bytes - ds.file_size_bytes)
            # 若该表原是 active，回退到最近剩余表的 active 并 reload
            if session.active_dataset_id == dataset_id:
                if session.datasets:
                    # 选 uploaded_at 最大的剩余表
                    next_id = max(session.datasets.values(),
                                  key=lambda d: d.uploaded_at).dataset_id
                    session.active_dataset_id = next_id
                    session.analysis_packages = dict(session.dataset_packages.get(next_id, {}))
                    nd = session.datasets[next_id]
                    if nd.df is None:
                        if nd.original_path and os.path.exists(nd.original_path):
                            try:
                                nd.df = pd.read_pickle(nd.original_path)
                            except Exception:
                                if nd.df_original is not None:
                                    nd.df = nd.df_original
                        elif nd.df_original is not None:
                            nd.df = nd.df_original
                else:
                    session.active_dataset_id = None
                    session.analysis_packages = {}
            session.last_access = time.time()
            crud.delete_dataset(session_id, dataset_id)
            self._persist_session(session_id)
            return True

    def set_dataset_packages(self, session_id: str, dataset_id: str, package_map: Dict[str, Any]):
        """按 dataset_id 分桶保存分析产物（修复三）"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session
            session.dataset_packages[dataset_id] = package_map
            # 修复三闭环：若正好是 active 数据集，同步进 session.analysis_packages 供看板/报告直接读取
            if session.active_dataset_id == dataset_id:
                session.analysis_packages = dict(package_map)
            session.last_access = time.time()
            self._persist_session(session_id)

    def get_dataset_packages(self, session_id: str, dataset_id: str) -> Dict[str, Any]:
        """获取指定数据集的分析产物（分桶）"""
        session = self.get_session(session_id)
        if session is None:
            return {}
        return dict(_safe_dict(session.dataset_packages).get(dataset_id, {}))

    def update_data(self, session_id: str, df: pd.DataFrame):
        """更新当前 DataFrame（清洗后），如果 session 不存在则自动创建"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session
            session.df = df.copy()
            session.last_access = time.time()
    
    def add_cleaning_step(self, session_id: str, step: Dict):
        """添加清洗记录"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session
            session.cleaning_history.append(step)
            session.last_access = time.time()
            self._persist_session(session_id)
    
    def get_cleaning_history(self, session_id: str) -> List[Dict]:
        """获取清洗历史"""
        session = self.get_session(session_id)
        return session.cleaning_history if session else []

    # ===== 任务1：AI 会话多轮聊天历史 =====
    def append_history(self, session_id: str, role: str, content: str,
                        artifacts_tags: Optional[List[str]] = None,
                        is_summarized: bool = False) -> None:
        """追加一条会话聊天记录（供 agent ReAct 循环 / 路由调用，维护多轮上下文）。

        role 取值建议："user" / "assistant" / "tool"。
        artifacts_tags: 该轮产出的轻量标签列表（如 ["图表:营收环形图"]），
                       用于避免把大段产物原文塞进上下文导致 AI 上下文膨胀，默认空列表。
        is_summarized: 该条是否已被滚动摘要压缩过，默认 False（供后续"记忆瘦身"识别新笔记）。
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session
            session.chat_history.append({
                "role": role,
                "content": content,
                "ts": time.time(),
                "artifacts_tags": artifacts_tags or [],
                "is_summarized": is_summarized,
            })
            session.last_access = time.time()
            self._persist_session(session_id)

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话聊天历史（按时间正序），供 agent 拼装 ctx["history"]。"""
        session = self.get_session(session_id)
        return session.chat_history if session else []

    # ===== 任务1：用户偏好档案卡（防偏好锁死）=====
    def set_user_preference(self, session_id: str, key: str, value: Any) -> None:
        """写入/覆盖一项用户偏好（写入即覆盖，保证偏好跟随最新意图，不被锁死）。

        约定：偏好只记录"用户最近的真实业务意图"（如最近一次明确的业务问题），
        不干预"分析任务选型"的细节。例：set_user_preference(sid, "last_business_question", "哪个品卖得不好")
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session
            session.user_preferences[key] = value
            session.last_access = time.time()
            self._persist_session(session_id)

    def get_user_preferences(self, session_id: str) -> Dict[str, Any]:
        """获取全部用户偏好（agent 自行 .get(key) 查单项；用户最新指令优先于此处值）。"""
        session = self.get_session(session_id)
        return session.user_preferences if session else {}
    
    def set_api_key(self, session_id: str, api_key: str):
        """设置 API Key"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session
            session.api_key = api_key
            session.last_access = time.time()
            self._persist_session(session_id)
    
    def get_api_key(self, session_id: str) -> str:
        """获取 API Key"""
        session = self.get_session(session_id)
        return session.api_key if session else ""

    def set_api_config(self, session_id: str, api_key: str, ai_provider: str,
                       custom_model: str, custom_base_url: str):
        """设置整套 AI 配置（api_key/ai_provider/custom_model/custom_base_url）并落库"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session
            session.api_key = api_key
            session.ai_provider = ai_provider
            session.custom_model = custom_model
            session.custom_base_url = custom_base_url
            session.last_access = time.time()
            self._persist_session(session_id)

    def get_api_config(self, session_id: str) -> Dict[str, str]:
        """获取整套 AI 配置，session 不存在时返回全空字符串"""
        session = self.get_session(session_id)
        if not session:
            return {"api_key": "", "ai_provider": "", "custom_model": "", "custom_base_url": ""}
        return {
            "api_key": session.api_key,
            "ai_provider": session.ai_provider,
            "custom_model": session.custom_model,
            "custom_base_url": session.custom_base_url,
        }

    def set_custom_title(self, session_id: str, title: str):
        """设置用户手动编辑的仪表盘标题"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session
            session.custom_title = title
            session.last_access = time.time()
            self._persist_session(session_id)

    def get_custom_title(self, session_id: str) -> str:
        """获取用户手动编辑的仪表盘标题"""
        session = self.get_session(session_id)
        return session.custom_title if session else ""

    def set_analysis_packages(self, session_id: str, packages: dict):
        """暂存分析结果（/analysis/run 后调用）"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session
            session.analysis_packages = packages
            # 闭环：把结果同时写进当前 active 数据集的桶，避免切换回来后丢失
            if session.active_dataset_id:
                session.dataset_packages[session.active_dataset_id] = packages
            session.last_access = time.time()
            self._persist_session(session_id)
    
    def push_undo_state(self, session_id: str):
        """保存当前状态到撤销栈（最多 20 步）"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.df is None:
                return
            session.df_undo_stack.append(session.df.copy())
            # 限制栈大小
            if len(session.df_undo_stack) > 20:
                session.df_undo_stack.pop(0)
            session.last_access = time.time()
            self._persist_session(session_id)

    def undo_last_action(self, session_id: str) -> Optional[pd.DataFrame]:
        """撤销上一步操作，返回恢复后的 DataFrame"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or len(session.df_undo_stack) == 0:
                return None
            prev_df = session.df_undo_stack.pop()
            session.df = prev_df.copy()
            session.last_access = time.time()
            self._persist_session(session_id)
            return session.df

    def get_undo_count(self, session_id: str) -> int:
        """获取可撤销步数"""
        session = self.get_session(session_id)
        return len(session.df_undo_stack) if session else 0

    def save_chart(self, session_id: str, chart: Dict[str, Any]):
        """保存图表到仪表盘收藏"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionData()
                self._sessions[session_id] = session
            chart["saved_at"] = time.time()
            session.saved_charts.append(chart)
            session.last_access = time.time()
            self._persist_session(session_id)

    def get_saved_charts(self, session_id: str) -> List[Dict[str, Any]]:
        """获取所有已保存的图表"""
        session = self.get_session(session_id)
        return session.saved_charts if session else []

    def delete_saved_chart(self, session_id: str, index: int) -> bool:
        """删除指定索引的已保存图表"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and 0 <= index < len(session.saved_charts):
                session.saved_charts.pop(index)
                session.last_access = time.time()
                self._persist_session(session_id)
                return True
            return False

    def clear_saved_charts(self, session_id: str):
        """清空所有已保存图表"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.saved_charts.clear()
                session.last_access = time.time()
                self._persist_session(session_id)

    # ===== V2 分析包操作 =====
    def save_packages(self, session_id: str, package_ids: List[str], dataset_id: Optional[str] = None):
        """聚合所有 dataset 分桶的包 + 兜底 analysis_packages，按 package_ids 复制进 saved_packages"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return
            # 聚合全部分桶的分析包（覆盖合表/多表 process-datasets 场景）
            all_pkgs: Dict[str, Any] = {}
            for bucket in session.dataset_packages.values():
                if isinstance(bucket, dict):
                    for pid, pkg in bucket.items():
                        all_pkgs[pid] = pkg
            # 兜底：同步并入 analysis_packages（老 /analysis/run 路径）
            if isinstance(getattr(session, 'analysis_packages', None), dict):
                for pid, pkg in session.analysis_packages.items():
                    all_pkgs.setdefault(pid, pkg)
            for pkg_id in package_ids:
                if pkg_id in all_pkgs:
                    src = all_pkgs[pkg_id]
                    pkg = dataclasses.asdict(src) if dataclasses.is_dataclass(src) else dict(src)
                    pkg["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    # ★ 标记包所属数据集，供报告/看板按当前激活数据集精准过滤。
                    #    优先使用前端显式传入的 dataset_id（解耦 session 状态），
                    #    回退 session.active_dataset_id（兼容旧调用方）。
                    effective_did = dataset_id or getattr(session, "active_dataset_id", None)
                    if effective_did:
                        pkg["dataset_id"] = effective_did
                    # 去重：同一 ID 不重复保存
                    if not any(p.get("id") == pkg_id for p in session.saved_packages):
                        session.saved_packages.append(pkg)
            session.last_access = time.time()
            self._persist_session(session_id)

    def get_saved_packages(self, session_id: str) -> List[Dict[str, Any]]:
        """获取所有已保存的分析包"""
        session = self.get_session(session_id)
        return session.saved_packages if session else []

    def get_saved_packages_full(self, session_id: str) -> List[Dict[str, Any]]:
        """获取已保存的分析包（含渲染后的 KPI/Table/Chart/Insight/Conclusion）

        使用 Renderer 层将 AnalysisPackage 的原始数据渲染为前端可消费格式。
        """
        from src.analysis_engine.package_render import render_package

        session = self.get_session(session_id)
        if not session:
            return []

        full_packages = []
        for pkg in session.saved_packages:
            full_packages.append(render_package(pkg))

        return full_packages

    def clear_data(self, session_id: str):
        """清除会话数据（同步删除落盘的原始文件 + SQLite 记录）；释放插槽后晋升队首"""
        with self._lock:
            self._remove_original_file(session_id)
            self._sessions.pop(session_id, None)
            crud.delete_session(session_id)
            self._promote_head()
    
    def _cleanup_sync(self):
        """清理过期会话（非线程安全，需在锁中调用）。

        同时处理限流相关释放：预约超时未上传的占槽空会话、
        排队票据超时，并在腾出插槽后晋升队首。
        """
        now = time.time()
        # 1) 过期会话（含已占插槽但整体超时的数据会话）
        expired = [
            sid for sid, sdata in self._sessions.items()
            if now - sdata.last_access > self._session_timeout
        ]
        for sid in expired:
            self._remove_original_file(sid)
            crud.delete_session(sid)
            del self._sessions[sid]
        # 2) 预约超时未上传（占槽空会话）：释放插槽，避免长期占槽
        for sid, sdata in list(self._sessions.items()):
            if sdata.holds_slot and sdata.df is None and (now - sdata.reserved_at) > self._RESERVE_TTL:
                self._remove_original_file(sid)
                crud.delete_session(sid)
                del self._sessions[sid]
        # 2.5) 已加载数据但空闲超时的插槽：分层释放，避免「刷新后数据悄悄没了」
        # - 已保存过图表/分析的会话（有持久价值）：超过 _saved_idle_timeout 仅释放内存(df=None)
        #   + 释放插槽(holds_slot=False)，但保留落盘原始文件（回来可从落盘 reload）与 SQLite 配置；
        #   严格不碰 active_dataset_id，保证用户回来仍能重新上传/清洗。
        # - 未保存会话（无持久价值）：超过 _slot_idle_timeout 走原 _release_slot_inner
        #   （释放内存+删落盘+让槽），彻底清。
        for sid, sdata in list(self._sessions.items()):
            if not sdata.holds_slot or sdata.df is None:
                continue
            idle = now - sdata.last_access
            has_saved = bool(sdata.saved_charts or sdata.saved_packages)
            if has_saved:
                if idle > self._saved_idle_timeout:
                    sdata.df = None            # 仅释放内存
                    sdata.holds_slot = False   # 让槽，不永久占槽
                    # 注意：不调 _remove_original_file，保留落盘；不碰 active_dataset_id
            else:
                if idle > self._slot_idle_timeout:
                    self._release_slot_inner(sid)  # 原逻辑：释放内存+删落盘+让槽
        # 3) 排队票据超时丢弃
        self._queue = [it for it in self._queue if now - it["created_at"] <= self._QUEUE_TTL]
        # 4) 腾出插槽后晋升队首
        self._promote_head()
    
    def cleanup(self):
        """清理过期会话（线程安全）"""
        with self._lock:
            self._cleanup_sync()

    def _start_background_cleanup(self):
        """启动后台守护线程，定时主动回收过期会话。

        弥补 request 触发的惰性清理（结论五）：无此后台线程时，过期会话要等到
        「下次请求触发 _cleanup_sync」或「超 max_sessions」才删，可能远超时 timeout。
        线程仅在持锁调用 cleanup()，而 _cleanup_sync 内不二次加锁、_remove_original_file
        亦设计为锁内调用，故无死锁风险。Python 引用语义保证清理瞬间正在使用的会话对象不被释放。
        """
        def _loop():
            while True:
                time.sleep(self._cleanup_interval)
                try:
                    self.cleanup()
                except Exception:
                    # 单轮清理异常不影响后续周期
                    pass
        t = Thread(target=_loop, name="session-cleanup", daemon=True)
        t.start()


def _safe_dict(v, default=None):
    """会话字段 None 兜底，避免遍历 None 触发 500（脏会话 datasets 为 JSON null 时）。"""
    if v is None:
        return default if default is not None else {}
    return v


# 全局单例
manager = SessionManager()
