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

_local = threading.local()
_lock = threading.RLock()

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
