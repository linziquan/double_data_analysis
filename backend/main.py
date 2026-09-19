"""
DataMind AI - FastAPI 后端入口
提供 RESTful API 接口供 React 前端调用
"""
import os
import sys
import time
import traceback

# 添加项目根目录到 sys.path，以便导入现有模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 加载 .env 环境变量（优先于 config 导入，本地开发可覆盖默认值，生产环境无 .env 文件自动用默认值）
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, ".env"))
# 同时加载 backend/.env（Agnes Key 等后端专属密钥放在此处，已被 gitignore 忽略，不进仓库）
load_dotenv(os.path.join(project_root, "backend", ".env"))

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, Response

# 导入路由
from backend.routers import upload, data, clean, chart, dashboard, insights, report, analysis, chat, auth, history
from backend.services.session_manager import manager
from backend.db.connection import init_db, get_connection, SQLITECLOUD_URL

# ===== 强制 UTF-8 编码，避免 Windows 环境下 print() 中文报错 =====
import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    try:
        _sys.stdout.reconfigure(encoding='utf-8')
        _sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

app = FastAPI(
    title="DataMind AI",
    description="数据分析智能体 API",
    version="1.0.0",
)

@app.on_event("startup")
def _startup_init_db():
    """启动时确保 SQLite 表结构存在（幂等建表）。"""
    import logging as _logging
    _logger = _logging.getLogger("uvicorn.error")
    try:
        init_db()
    except Exception as exc:  # 建表失败不应阻断启动，但需记录
        _logger.error(f"数据库初始化失败: {exc}", exc_info=True)

    # 已知脏会话一次性清理：旧版（建库前 bug 版）把 AnalysisPackage 对象经
    # json.dumps(default=str) 序列化成了字符串落库，重启读回后无法还原，
    # 既刷屏又导致分析列表为空。该会话的已有分析产物已确认丢弃，用户需重新上传。
    try:
        from backend.db import crud as _crud
        _DIRTY_SESSIONS = ["fd22be77-80d3-4867-9619-5e6259ea8826"]
        for _sid in _DIRTY_SESSIONS:
            _crud.delete_session(_sid)
        if _DIRTY_SESSIONS:
            _logger.info(
                f"已清理 {len(_DIRTY_SESSIONS)} 个历史脏会话的 SQLite state"
            )
    except Exception as exc:
        _logger.warning(f"清理历史脏会话失败(可忽略): {exc}")

    # 冷启动清理：仅释放游客（user_id IS NULL）数据，保留登录用户及其落盘 pkl。
    # 与用户约定一致：重启后端即释放游客数据（数据集/分析包/已保存图表/落盘 pkl），
    # 但登录用户的数据归属已归集，不应被误删（clear_guest_data 内部先收路径再删行最后删文件）。
    try:
        from backend.db import crud as _crud
        _deleted = _crud.clear_guest_data()
        _logger.info(f"已清理游客上传数据（{_deleted} 个数据集），登录用户数据已保留")
    except Exception as exc:
        _logger.warning(f"冷启动清空游客数据失败(可忽略): {exc}")

    # 进程内心跳：每 60s 探活一次共享连接，确保 SQLite Cloud 连接保持温热，
    # 避免免费层空闲断开导致下次请求重建（约 20s）。与 GitHub Actions 保活互补
    # （后者负责让 Render 实例不休眠，前者负责让数据库连接不冷）。
    try:
        import threading as _threading

        def _db_heartbeat():
            while True:
                _threading.Event().wait(60)
                try:
                    c = get_connection()
                    c.execute("SELECT 1").fetchone()
                except Exception as _he:
                    _logger.warning(f"DB 心跳失败(将在下次请求重建连接): {_he}")

        _hb = _threading.Thread(target=_db_heartbeat, name="db-heartbeat", daemon=True)
        _hb.start()
    except Exception as exc:
        _logger.warning(f"启动 DB 心跳线程失败(可忽略): {exc}")


# CORS 配置 - 演示阶段允许所有来源（生产环境应限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # allow_origins=["*"] 时必须为 False
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(upload.router, prefix="/api", tags=["数据上传"])
app.include_router(data.router, prefix="/api", tags=["数据操作"])
app.include_router(clean.router, prefix="/api", tags=["数据清洗"])
app.include_router(chart.router, prefix="/api", tags=["图表生成"])
app.include_router(dashboard.router, prefix="/api", tags=["仪表盘"])
app.include_router(insights.router, prefix="/api", tags=["AI 洞察"])
app.include_router(report.router, prefix="/api", tags=["报告生成"])
app.include_router(analysis.router, prefix="/api", tags=["分析执行"])
app.include_router(chat.router, prefix="/api", tags=["聊天对话"])
app.include_router(auth.router, tags=["认证"])
app.include_router(history.router, tags=["历史记录"])


# 全局异常处理器：捕获所有未处理的异常，返回详细错误信息
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常捕获：避免 500 时前端只看到 Network Error"""
    tb = traceback.format_exc()
    import logging as _logging; _logging.getLogger("uvicorn.error").error(f"{exc.__class__.__name__}: {exc}", exc_info=True)
    # traceback logged via logging above
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"{exc.__class__.__name__}: {str(exc)}",
            "traceback": tb if os.getenv("DEBUG") == "1" else None,
        }
    )


@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/db-ping")
async def db_ping():
    """轻量数据库探活：供外部保活定时任务（如 GitHub Actions 每 5 分钟 ping 一次）调用。

    作用：让 Render 免费实例保持唤醒，并周期性复用 SQLite Cloud 连接，
    使真实登录时基本不触发冷连接重建，登录更快更稳。
    返回 ok / 数据库类型 / 本次探活耗时（毫秒）。"""
    t0 = time.time()
    try:
        conn = get_connection()
        conn.execute("SELECT 1").fetchone()
        return {
            "ok": True,
            "db": "cloud" if SQLITECLOUD_URL else "local",
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as exc:
        return {
            "ok": False,
            "db": "cloud" if SQLITECLOUD_URL else "local",
            "error": f"{exc.__class__.__name__}: {exc}",
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }



@app.get("/api/session/new")
async def new_session():
    """创建新会话"""
    session_id = manager.create_session()
    return {"session_id": session_id, "success": True}


@app.post("/api/session/clear")
async def clear_session(session_id: str = Query(...)):
    """结束会话：级联清空该会话全部数据（落盘文件 + SQLite 数据集/分析包 + 已保存图表），释放插槽"""
    manager.clear_data(session_id)
    return {"status": "ok"}


@app.post("/api/session/page")
async def set_session_page(session_id: str = Query(...), page: str = Query(...)):
    """记录会话当前所在页面，供历史恢复时智能跳转。游客态（无归属）亦允许，仅作轻量标记。"""
    manager.set_current_page(session_id, page)
    return {"status": "ok"}


# 前端构建产物目录（Render 部署时随仓库提交 frontend/dist，由后端直接托管）
# 兼容不同部署目录结构：在多个候选路径中查找 frontend/dist
def _resolve_frontend_dist():
    """在多个候选路径中寻找 frontend/dist，兼容不同部署目录结构"""
    candidates = [
        os.path.join(project_root, "frontend", "dist"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist"),
        os.path.join(os.getcwd(), "frontend", "dist"),
        os.path.join(os.path.dirname(os.getcwd()), "frontend", "dist"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    # 兜底：返回标准路径（即使不存在，main.py 会回退到 JSON）
    return os.path.join(project_root, "frontend", "dist")

FRONTEND_DIST = _resolve_frontend_dist()


@app.get("/")
async def root():
    """根路径：部署时返回前端 SPA 页面，本地未构建时返回健康检查 JSON"""
    index_html = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.isfile(index_html):
        # index.html 禁止 CDN 缓存，确保用户总能拿到最新版本
        return FileResponse(index_html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return {"status": "ok", "version": "1.0.0"}


# SPA fallback：非 /api 的 GET 请求先尝试返回对应静态资源，找不到则回退到 index.html
# 仅在构建产物存在时注册（本地开发走 Vite dev server，无需此后端托管）
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if not os.path.isdir(FRONTEND_DIST):
        return {"status": "ok", "version": "1.0.0"}
    if full_path.startswith("api/"):
        # 真实 API 路由已在上面注册；此处兜底返回 JSON，避免 SPA 回退吞掉 /api 请求
        return {"status": "ok", "version": "1.0.0"}
    file_path = os.path.join(FRONTEND_DIST, full_path)
    if os.path.isfile(file_path):
        # 带 hash 的静态资源（JS/CSS/图片）缓存 1 年；其他文件不缓存
        ext = os.path.splitext(full_path)[1].lower()
        if ext in ('.js', '.css', '.woff', '.woff2', '.ttf', '.svg', '.png', '.jpg', '.ico'):
            return FileResponse(file_path, headers={"Cache-Control": "public, max-age=31536000, immutable"})
        return FileResponse(file_path, headers={"Cache-Control": "no-cache"})
    return FileResponse(os.path.join(FRONTEND_DIST, "index.html"), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
