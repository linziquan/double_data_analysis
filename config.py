"""
配置文件 - 数据分析智能体
用户需要输入自己的 DeepSeek API Key 才能使用 AI 功能
"""
import os

# DeepSeek API 配置
DEEPSEEK_API_URL = "https://api.deepseek.com"

# 默认模型
DEFAULT_MODEL = "deepseek-chat"

# 应用配置
APP_TITLE = "DataMind AI - 数据分析智能体"
APP_ICON = "📊"

# 文件上传配置
# 通过环境变量 MAX_UPLOAD_SIZE_MB 控制，默认 30MB
# 本地开发在 .env 文件中设置 MAX_UPLOAD_SIZE_MB=5120 可覆盖到 5GB，该文件为 git ignore 不上线
MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 30))
MAX_FILE_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
SUPPORTED_FORMATS = ["csv", "xlsx", "xls", "json", "db", "sqlite"]
# 单会话累计文件额度（所有已上传文件字节之和 ≤ 此值）；上线固定 30MB 防 OOM
QUOTA_BYTES = MAX_FILE_SIZE_BYTES

# 图表配色（清新浅绿渐变方案）
CHART_COLORS = ["#9FD8C8", "#5CB8A2", "#5A7C74", "#94B0A9", "#C7E6DF", "#2A4A43"]

# AI 配置
AI_TEMPERATURE = 0.3
AI_MAX_TOKENS = 2048

# ============================================================
# JWT 鉴权配置
# ============================================================
# 密钥优先从环境变量 JWT_SECRET 读取（生产必备）。
# 缺失时在开发期随机生成并打印警告（不阻断启动），但不应用于生产。
import secrets as _secrets
import logging as _logging

_jwt_secret = os.environ.get("JWT_SECRET")
if not _jwt_secret:
    _jwt_secret = "dev-secret-fixed-do-not-use-in-prod"
    _logging.getLogger(__name__).warning(
        "JWT_SECRET 未设置，使用本地开发固定密钥。生产环境请通过环境变量 JWT_SECRET 配置固定密钥。"
    )
JWT_SECRET = _jwt_secret
