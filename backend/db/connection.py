"""
SQLite 连接管理。

设计要点：
- 数据库路径从环境变量 DB_PATH 读取（默认 data/app.db），不在代码写死，支持解耦部署时换路径/换服务器。
- 单连接 + 线程锁保护，兼容 SessionManager 现有的多线程访问模型（已用 RLock）。
- 首次调用 get_connection 时自动按 schema.sql 建表（幂等，IF NOT EXISTS）。
- 所有 SQL 集中在 schema.sql 与 crud.py，本文件只管连接生命周期。
"""
import os
import sqlite3
import threading
import logging

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

_local = threading.local()
_lock = threading.RLock()


def _ensure_dir() -> None:
    parent = os.path.dirname(DB_PATH)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _add_user_columns(conn: sqlite3.Connection) -> None:
    """幂等为三张表补充 user_id 列（兼容升级前的旧库）。

    必须在已持有 conn 时调用（不走 get_connection，避免与 _run_schema 相互递归）。
    """
    for table in ("sessions", "datasets", "analysis_packages"):
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_user ON {table}(user_id)")
        except sqlite3.OperationalError:
            # 列已存在则忽略
            conn.rollback()


def _run_schema(conn: sqlite3.Connection) -> None:
    """执行 schema.sql 建表（幂等）。"""
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema.sql 未找到: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    # 幂等为三表补充 user_id 列（兼容升级前的旧库），使用同一 conn 避免递归
    _add_user_columns(conn)


def get_connection() -> sqlite3.Connection:
    """返回当前线程的 SQLite 连接（懒初始化 + 自动建表）。

    使用线程局部存储，避免多线程共享同一连接导致的 sqlite 线程错误
    （sqlitecloud 驱动的 threadsafety 同样为 1，因此这里保持不变）；
    写操作由 SessionManager 的 RLock 串行化，连接层本身不引入额外并发模型。

    数据源按环境二选一：
      - 配了 SQLITECLOUD_URL → 连接 SQLite Cloud（Render 等无持久磁盘环境使用）
      - 未配置              → 本地 SQLite 文件（默认，本地开发）
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn
    if SQLITECLOUD_URL:
        # 延后导入：未使用云端时不强制安装该依赖
        import sqlitecloud

        # 返回的行必须支持按列名取值，对应 crud.py 中的 row["xxx"] 写法；
        # 注意要用 sqlitecloud.Row 而非 sqlite3.Row（后者无法跨驱动实例化）。
        conn = sqlitecloud.connect(SQLITECLOUD_URL)
        conn.row_factory = sqlitecloud.Row
    else:
        _ensure_dir()
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
    # 外键约束开启（schema 中使用了 FK 语义，便于分发后别人理解关系）
    conn.execute("PRAGMA foreign_keys = ON")
    with _lock:
        _run_schema(conn)
    _local.conn = conn
    return conn


def close_connection() -> None:
    """关闭当前线程连接（进程退出或测试清理时调用）。"""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None


def init_db() -> None:
    """显式初始化数据库（建表）。供启动入口或 init_db.py 调用。"""
    conn = get_connection()
    with _lock:
        _run_schema(conn)
    logger.info("SQLite 数据库已初始化: %s", DB_PATH)
