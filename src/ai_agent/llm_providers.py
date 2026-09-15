"""LLM 服务商 → (base_url, 默认 model) 映射与解析（SSOT）。

- 必须与前端 frontend/src/contexts/DataContext.tsx 的 AI_PROVIDERS 保持同步；
- 被 backend/routers/chat.py（聊天主链路）与 src/tools_registry.py（报告等工具内部的
  LLM 调用）共用，确保「工具内部的 LLM 调用」跟随用户所选服务商与 Key，
  而不是写死 Agnes —— 写死会导致用户配了 deepseek 却仍用 Agnes 而 401、报告降级。
"""
from typing import Dict, Optional

LLM_PROVIDERS: Dict[str, Dict[str, str]] = {
    "ppio":      {"base_url": "https://api.ppio.ai/v1",          "model": "deepseek-chat"},
    "deepseek":  {"base_url": "https://api.deepseek.com",        "model": "deepseek-chat"},
    "qwen":      {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen3.7-plus"},
    "zhipu":     {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash"},
    "moonshot":  {"base_url": "https://api.moonshot.cn/v1",       "model": "moonshot-v1-8k"},
    "openai":    {"base_url": "https://api.openai.com/v1",        "model": "gpt-4o-mini"},
    "agnes":     {"base_url": "https://apihub.agnes-ai.com/v1",   "model": "agnes-2.0-flash"},
    "opencodex": {"base_url": "https://openrouter.ai/api/v1",      "model": "stealth/ox-alpha"},
}


def resolve_llm_config(
    ai_provider: Optional[str],
    custom_model: Optional[str] = None,
    custom_base_url: Optional[str] = None,
) -> Dict[str, str]:
    """按服务商解析本次调用使用的 (model, base_url)。

    优先级：custom_model > 服务商默认 model；custom_base_url > 服务商默认 base_url；
    ai_provider 未识别时退回 agnes（与 DataAnalysisAgent 构造默认值一致）。
    """
    provider = (ai_provider or "agnes").lower()
    preset = LLM_PROVIDERS.get(provider, LLM_PROVIDERS["agnes"])
    return {
        "model": (custom_model or "").strip() or preset["model"],
        "base_url": (custom_base_url or "").strip() or preset["base_url"],
    }
