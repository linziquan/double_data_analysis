-- ============================================================
-- DataMind AI 数据库结构定义（SQLite）
-- ============================================================
-- 用途：替代原"内存 + 临时 pickle"存储，使 session / 上传数据 / 分析结果
--       在后端重启或云实例休眠后不丢失，并可作为独立组件分发给他人。
--
-- 分发说明：别人拿到本 schema.sql + 任意 .db 文件（或空库），执行本脚本即可
--           重建全部表结构；无需安装额外数据库软件，Python 标准库 sqlite3 即可读写。
--
-- 建表幂等：全部使用 IF NOT EXISTS，可重复执行。
-- ============================================================

-- 会话表：存储单个会话的可序列化状态（不含 DataFrame 本体）。
-- DataFrame 体积可能很大（本地支持 1GB），不强行塞 BLOB，
-- 而是保留"落盘 pickle 文件"机制，路径索引在 datasets 表。
-- state_json 含：api_key / custom_title / cleaning_history / analysis_history /
--                saved_charts / analysis_packages / saved_packages / df_undo_stack /
--                dataset_packages 索引 / active_dataset_id / 各类时间戳标记。
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    state_json   TEXT NOT NULL,            -- 会话可序列化状态（JSON）
    created_at   REAL NOT NULL,            -- 创建时间戳
    last_access  REAL NOT NULL             -- 最后访问时间戳（用于过期清理判断）
);

-- 数据集表：每个上传的报表对应一行。DataFrame 落盘为 pickle 文件，
-- 这里只记录其持久化路径（data/ 目录，已加入 .gitignore），重启后按路径 reload。
-- meta_json 含：file_name / file_size_bytes / rows / columns / column_info /
--               preview / is_merged / sources / merge_keys / uploaded_at。
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id   TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    meta_json    TEXT NOT NULL,            -- 数据集元信息（JSON）
    original_path TEXT NOT NULL,           -- DataFrame pickle 持久化路径
    is_active    INTEGER NOT NULL DEFAULT 0,-- 是否为该会话当前 active 数据集
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_datasets_session ON datasets(session_id);

-- 分析包表：AnalysisPackage 序列化后存储（dataclasses.asdict -> JSON）。
-- 一个数据集可对应多个分析包（package_id 唯一）。
CREATE TABLE IF NOT EXISTS analysis_packages (
    package_id   TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    dataset_id   TEXT NOT NULL,
    payload_json TEXT NOT NULL,            -- AnalysisPackage 完整 JSON
    saved_at     TEXT,                     -- 用户保存时间戳（可空）
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_packages_session ON analysis_packages(session_id);
CREATE INDEX IF NOT EXISTS idx_packages_dataset ON analysis_packages(dataset_id);

-- 说明：saved_packages（用户已保存的分析包列表）已并入 sessions.state_json，
--       无需独立表；如需独立审计可后续拆分，但本版遵循最小改动原则。

-- ============================================================
-- 用户账户表（登录系统核心）
-- ============================================================
-- token_version：改密 / 退出时 +1，使旧 JWT 立即失效（无需黑名单）。
-- storage_used / dataset_limit：P2 配额地基（P1 仅落列，dataset_limit 默认 10）。
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username       TEXT UNIQUE NOT NULL,
    password_hash  TEXT NOT NULL,
    token_version  INTEGER DEFAULT 0,
    storage_used   INTEGER DEFAULT 0,
    dataset_limit  INTEGER DEFAULT 50,
    created_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- 三表 user_id 列（可空 TEXT）：用于数据归属归集。
-- 通过 crud._ensure_user_columns() 幂等 ALTER 添加（兼容旧库），
-- 写入时一律经 crud.to_user_id_str 与 users.id 对齐，避免 int→TEXT 隐式转换污染。

-- ============================================================
-- 收藏 / 分组表（P2：用户对个人分析包收藏与分组管理）
-- ============================================================
-- 关联：user_id(users.id) + package_id(analysis_packages.package_id)，每个用户对同一
--       分析包最多一条收藏记录（UNIQUE 约束）。
-- is_starred：是否已收藏（0/1）。取消收藏时置 0 而非删除行，方便保留历史分组/重命名。
-- display_name：用户对该分析包的自定义显示名（覆盖原始标题），可空。
-- group_name：所属分组名，默认「默认分组」。
-- sort_order：同组内排序权重（越小越靠前）。
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
);
CREATE INDEX IF NOT EXISTS idx_favorites_user ON favorites(user_id);
CREATE INDEX IF NOT EXISTS idx_favorites_user_pkg ON favorites(user_id, package_id);

-- ============================================================
-- 分享链接表（P2：用户将分析包生成公开只读分享）
-- ============================================================
-- share_id：对外暴露的短标识（非自增，避免被遍历猜测），后端用短随机串生成。
-- package_id：被分享的分析包（analysis_packages.package_id）。
-- user_id：分享者（users.id）。
-- expire_at：过期时间戳（REAL，秒），可空表示永久有效。
-- 鉴权：GET /api/shared/{share_id} 无需登录即可读取（公开只读），其余写操作需登录。
CREATE TABLE IF NOT EXISTS shares (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    share_id   TEXT UNIQUE NOT NULL,
    package_id TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    created_at REAL NOT NULL,
    expire_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_shares_share_id ON shares(share_id);
CREATE INDEX IF NOT EXISTS idx_shares_user ON shares(user_id);

