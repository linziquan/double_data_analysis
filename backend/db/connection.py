"""
SQLite 连接管理。

设计要点：
- 数据库路径从环境变量 DB_PATH 读取（默认 data/app.db），不在代码写死，支持解耦部署时换路径/换服务器。
- 单连接 + 线程锁保护，兼容 SessionManager 现有的多线程访问模型（已用 RLock）。
- 首次调用 get_connection 时自动按 schema.sql 建表（幂等，IF NOT EXISTS）。
- 所有 SQL 集中在 schema.sql 与 crud.py，本文件只管连接生命周期。
- 双驱动兼容：本地走标准库 sqlite3，配置 SQLITECLOUD_URL 时走 sqlitecloud 驱动。
  两个驱动并非完全等价（云端不支持 executescript、异常类也不共享），
  因此本文件刻意只使用两者都有的接口，并用宽泛异常兜底（详见 _split_sql_statements
  与 _add_user_columns 的注释）。
"""
import os
import sqlite3
import threading
import logging
import time
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_default_db_path = os.path.join(_project_root, "data", "app.db")

# 从 .env 读取；若未设置则用默认 data/app.db（data/ 目录已加入 .gitignore，防大文件进仓库）
DB_PATH = os.environ.get("DB_PATH", _default_db_path)

# ===== 云端 SQLite（SQLite Cloud）=====
# Render 免费实例的磁盘是临时的：重新部署或实例休眠后 app.db 会被重置，
# 导致注册的账号与历史记录全部丢失。配置了 SQLITECLOUD_URL 时改用托管 SQLite，
# 未配置时行为与以前完全一致（本地文件），因此不会影响本地开发。
# 连接串形如：sqlitecloud://<host>:8860/<db>.sqlite?apikey=<apikey>
SQLITECLOUD_URL = os.environ.get("SQLITECLOUD_URL", "").strip()

# ===== 单例共享连接（替代线程局部）=====
# 原先每个线程各自维护一条到 SQLite Cloud 的连接，问题是：外部保活定时任务
# （GitHub Actions / 进程内心跳）只捂热了「一个线程」的连接，而登录等请求跑在
# 另一个线程池线程，每次都新建连接（SQLite Cloud 免费层建连约 20s）→ 登录慢。
# 改成「全进程唯一一条共享连接」，所有线程复用同一条温连接，建连开销只在
# 进程启动/重建时发生一次；并用全局锁串行化所有数据库操作（sqlitecloud
# threadsafety=1，连接本身不可跨线程并发使用）。
_shared_conn = None            # 共享连接（_LockedConn 包装）
_shared_last_used = 0.0       # 上次使用时间（用于空闲探活）
_db_lock = threading.RLock()  # 串行化所有数据库操作
_lock = threading.RLock()     # 建表 schema 专用（与 _db_lock 配合，顺序固定避免死锁）

# 进程级标记：schema 建表只需成功执行一次。
# 不设该死标记的话，每个线程新建连接都会重复跑一遍 schema.sql + 列检查，
# 在云端（每次 execute 都是一次网络往返）会明显拖慢首次请求。
_SCHEMA_READY = False


def _ensure_dir() -> None:
    parent = os.path.dirname(DB_PATH)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _split_sql_statements(script: str) -> List[str]:
    """把 schema.sql 文本拆成一条条可独立执行的 SQL 语句。

    为什么不直接用 conn.executescript()：本地 sqlite3 支持它，但云端驱动
    sqlitecloud 明确不支持（其 Connection.executescript 直接抛
    SQLiteCloudNotSupportedError），一旦用云端库就会在「首次建表」时崩溃。
    因此这里统一采用「先去 -- 行注释，再按 ; 拆分」，两种驱动都能逐条 execute。

    去注释是为了避免两种坑：注释里带分号被误切、以及纯注释片段被当成语句执行。
    schema.sql 中不含带分号的字符串字面量，故简单拆分即可。
    """
    lines: List[str] = []
    for line in script.splitlines():
        pos = line.find("--")
        lines.append(line[:pos] if pos >= 0 else line)
    return [stmt.strip() for stmt in "\n".join(lines).split(";") if stmt.strip()]


def _table_columns(conn: sqlite3.Connection, table: str) -> Optional[Set[str]]:
    """读取某表的列名集合；驱动不支持 PRAGMA 时返回 None（表示「未知」）。"""
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except Exception as e:
        logger.debug("PRAGMA table_info(%s) 不可用: %s", table, e)
        return None
    cols: Set[str] = set()
    for row in rows:
        try:
            cols.add(str(row["name"]))
        except Exception:
            # 兜底：PRAGMA 结果第 2 列即列名
            cols.add(str(row[1]))
    return cols


def _add_user_columns(conn: sqlite3.Connection) -> None:
    """幂等为三张表补充 user_id 列与索引（兼容升级前的旧库）。

    必须在已持有 conn 时调用（不走 get_connection，避免与 _run_schema 相互递归）。

    异常刻意捕获宽泛的 Exception：本地驱动抛 sqlite3.OperationalError，
    云端驱动抛自己的 SQLiteCloudOperationalError，两者没有继承关系；
    只捕获 sqlite3 的异常会让「列已存在」在云端变成启动失败。
    """
    for table in ("sessions", "datasets", "analysis_packages"):
        cols = _table_columns(conn, table)
        # PRAGMA 不可用（cols 为 None）时无法预判，只能尝试 ALTER 并容忍失败
        if cols is None or "user_id" not in cols:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT")
            except Exception as e:
                logger.debug("ALTER %s.user_id 失败（多半是列已存在）: %s", table, e)
                try:
                    conn.rollback()
                except Exception:
                    pass
        try:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_user ON {table}(user_id)")
        except Exception as e:
            logger.debug("为 %s(user_id) 建索引失败: %s", table, e)


def _run_schema(conn: sqlite3.Connection) -> None:
    """执行 schema.sql 建表（幂等；进程内只成功执行一次，避免每次建连接都跑脚本）。"""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema.sql 未找到: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        statements = _split_sql_statements(f.read())
    for stmt in statements:
        conn.execute(stmt)
    conn.commit()
    # 幂等为三表补充 user_id 列（兼容升级前的旧库），使用同一 conn 避免递归
    _add_user_columns(conn)
    _SCHEMA_READY = True


# ===== 云端连接超时与存活管理 =====
# 关键背景：sqlitecloud 0.0.84 的原生 SQLiteCloudConfig 自带 connect_timeout
# （默认 30s）与 timeout（默认 0，即操作无超时）。当前代码 connect(SQLITECLOUD_URL)
# 传的是字符串 DSN，SDK 会用 SQLiteCloudConfig(dsn) 重新解析并忽略外部 config 参数，
# 因此超时参数必须写进 DSN 查询串才生效。timeout=0 会让写操作在抖动链路上无限挂起，
# 超过 Render 代理(~50s)后被掐断成前端 Network Error；这里显式给一个上限。
_CLOUD_CONNECT_TIMEOUT = 30   # 建连上限（秒），贴近但不超 50s 代理掐断阈值
_CLOUD_SOCKET_TIMEOUT = 15    # 单次操作（含探活 SELECT 1）上限（秒）
_CLOUD_IDLE_MAX = 30          # 连接最大空闲秒数，超过则复用时先探活


def _inject_cloud_timeouts(dsn: str) -> str:
    """向 SQLite Cloud DSN 查询串注入 connect_timeout/timeout（已设置则保留原值）。

    仅在 DSN 未显式包含这两个 key 时追加，便于在 Render 环境变量里手动覆盖。
    """
    has_connect = "connect_timeout" in dsn
    has_timeout = "timeout" in dsn
    if has_connect and has_timeout:
        return dsn
    sep = "&" if "?" in dsn else "?"
    added = []
    if not has_connect:
        added.append(f"connect_timeout={_CLOUD_CONNECT_TIMEOUT}")
    if not has_timeout:
        added.append(f"timeout={_CLOUD_SOCKET_TIMEOUT}")
    return dsn + sep + "&".join(added)


class _LockedCursor:
    """游标代理：fetch 类方法在全局锁内执行，兼容 sqlitecloud / sqlite3 游标。"""

    def __init__(self, cursor, lock):
        object.__setattr__(self, "_cursor", cursor)
        object.__setattr__(self, "_lock", lock)

    def fetchone(self):
        with self._lock:
            return self._cursor.fetchone()

    def fetchall(self):
        with self._lock:
            return self._cursor.fetchall()

    def fetchmany(self, *args, **kwargs):
        with self._lock:
            return self._cursor.fetchmany(*args, **kwargs)

    def __iter__(self):
        with self._lock:
            return iter(self._cursor)

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _LockedConn:
    """连接代理：所有数据库操作在全局锁内串行执行，使单条共享连接可安全跨线程复用。"""

    def __init__(self, conn, lock):
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_lock", lock)

    def execute(self, *args, **kwargs):
        with self._lock:
            return _LockedCursor(self._conn.execute(*args, **kwargs), self._lock)

    def executemany(self, *args, **kwargs):
        with self._lock:
            return _LockedCursor(self._conn.executemany(*args, **kwargs), self._lock)

    def commit(self):
        with self._lock:
            return self._conn.commit()

    def rollback(self):
        with self._lock:
            return self._conn.rollback()

    def close(self):
        with self._lock:
            return self._conn.close()

    @property
    def row_factory(self):
        with self._lock:
            return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, value):
        with self._lock:
            self._conn.row_factory = value

    def __getattr__(self, name):
        with self._lock:
            return getattr(self._conn, name)


def _probe_alive(conn) -> bool:
    """轻量存活探针：受 DSN 的 timeout 约束，死连接会快速抛错而非挂起。"""
    try:
        conn.execute("SELECT 1").fetchone()
        return True
    except Exception as e:
        logger.debug("云端连接探活失败: %s", e)
        return False


def _create_connection():
    """新建并初始化一条共享连接（含建表）。调用方需已持有 _db_lock。"""
    global _shared_conn
    if SQLITECLOUD_URL:
        # 延后导入：未使用云端时不强制安装该依赖
        import sqlitecloud

        # 返回的行必须支持按列名取值，对应 crud.py 中的 row["xxx"] 写法；
        # 注意要用 sqlitecloud.Row 而非 sqlite3.Row（后者无法跨驱动实例化）。
        # 超时参数写进 DSN 查询串（见 _inject_cloud_timeouts 说明）。
        dsn = _inject_cloud_timeouts(SQLITECLOUD_URL)
        conn = sqlitecloud.connect(dsn)
        conn.row_factory = sqlitecloud.Row
        # 只打印主机/库名，绝不打 apikey
        logger.info("已连接 SQLite Cloud 托管库: %s", SQLITECLOUD_URL.split("?")[0])
    else:
        _ensure_dir()
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
    # 外键约束开启（schema 中使用了 FK 语义，便于分发后别人理解关系）。
    # 云端驱动/服务端未必支持该 PRAGMA，不阻断启动，仅记录日志。
    try:
        conn.execute("PRAGMA foreign_keys = ON")
    except Exception as e:
        logger.debug("PRAGMA foreign_keys 不受支持（已忽略）: %s", e)
    with _lock:
        _run_schema(conn)
    _shared_conn = _LockedConn(conn, _db_lock)
    return _shared_conn


def get_connection() -> sqlite3.Connection:
    """返回全进程共享的唯一 SQLite 连接（懒初始化 + 空闲探活自愈 + 全局串行化）。

    单例共享：所有线程（保活定时任务、登录请求、后台心跳）复用同一条连接，
    避免「每个线程各自建连」导致的登录冷连接慢（SQLite Cloud 免费层建连约 20s）。
    全局锁 _db_lock 串行化所有 execute/commit，兼容 sqlitecloud threadsafety=1
    （连接本身不可跨线程并发使用）。

    数据源按环境二选一：
      - 配了 SQLITECLOUD_URL → 连接 SQLite Cloud（Render 等无持久磁盘环境使用）
      - 未配置              → 本地 SQLite 文件（默认，本地开发）

    云端连接自愈：空闲超过 _CLOUD_IDLE_MAX 秒的共享连接可能已死
    （SQLite Cloud 免费层长链路间歇断连），复用时先发 SELECT 1 探活，
    失败则关闭重建，避免写操作时才暴露 "writing data"。建连开销因此只在
    进程启动/重建时发生一次，日常请求直接复用温连接。
    """
    global _shared_last_used
    with _db_lock:
        conn = _shared_conn
        if conn is not None and SQLITECLOUD_URL and (time.time() - _shared_last_used) > _CLOUD_IDLE_MAX:
            if not _probe_alive(conn):
                logger.info("云端连接空闲过久且探活失败，关闭并重建")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = None
        if conn is None:
            conn = _create_connection()
        _shared_last_used = time.time()
        return conn


def close_connection() -> None:
    """关闭共享连接（进程退出或测试清理时调用）。"""
    global _shared_conn, _shared_last_used
    with _db_lock:
        conn = _shared_conn
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        _shared_conn = None
        _shared_last_used = 0.0


def init_db() -> None:
    """显式初始化数据库（建表）。供启动入口或 init_db.py 调用。"""
    conn = get_connection()
    with _lock:
        _run_schema(conn)
    logger.info("SQLite 数据库已初始化: %s", DB_PATH)
