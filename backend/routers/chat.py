"""
聊天路由（智能体版）：将用户消息交给 DataAnalysisAgent 的 agentic_chat 做
function calling 循环，支持多轮 choice 选择、工具执行、清洗后自动分析。

链路：
上传即侦察 → 用户发消息 → agentic_chat（LLM 调工具直到给出最终回答）
→ 结构化响应 {kind, content, choices, tool_results, data_preview}
→ 前端渲染（text / choice 按钮 / 工具执行状态 / 数据预览）
"""
import asyncio
import traceback
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from backend.services.session_manager import manager
from src.ai_agent.agent import DataAnalysisAgent
from src.utils.json_serializer import sanitize_json

router = APIRouter()


def _sanitize_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """写回 session 前净化历史：剥离工具调用痕迹，只保留纯文字对话。

    保留：system / user / assistant（去掉 tool_calls 字段）消息。
    剥离：role=="tool" 的回灌消息，以及 assistant 消息里的 tool_calls 键。

    这样下一轮 agentic_chat 读 history 时，LLM 看不到上一轮的工具链，
    不会误判为"未完成任务"而重调工具（切断大类 B 死循环源）。
    注意：clean_data 体检态的弹框依赖 tool_results + await_choice 字段返回给前端，
    不依赖写回 history 的 tool 消息，故剥离不影响前端交互。
    """
    cleaned: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            continue  # 丢弃工具回灌消息
        if role == "assistant":
            # 复制并去掉 tool_calls 字段（保留 content 文字）
            item = {k: v for k, v in m.items() if k != "tool_calls"}
            cleaned.append(item)
        else:
            # system / user 原样保留
            cleaned.append(m)
    return cleaned


class ChatRequest(BaseModel):
    session_id: str
    message: str
    choice: Optional[str] = None   # 用户点击的清洗方案 id（多轮续接时带）
    # —— AI 模型配置（来自 API 配置页）：任一为空时后端沿用 .env 默认（Agnes） ——
    api_key: Optional[str] = None
    ai_provider: Optional[str] = None
    custom_model: Optional[str] = None
    custom_base_url: Optional[str] = None


# 合法清洗方法集合，必须与 src/tools_registry.py 的 _FIVE_METHODS_META[].method 及
# clean_data 工具 schema 的 enum 保持一致；新增清洗方法时需同步更新此处。
LEGAL_METHODS = {"fill_mean", "fill_median", "fill_mode", "fill_0"}


# AI 服务商映射已收敛到 src/ai_agent/llm_providers.py（SSOT），与报告等工具内部的
# LLM 调用共用，避免「聊天用 deepseek、报告却写死 Agnes」导致报告 401 降级。
# 该映射仍需与 frontend/src/contexts/DataContext.tsx 的 AI_PROVIDERS 保持同步。
from src.ai_agent.llm_providers import LLM_PROVIDERS as _LLM_PROVIDERS  # noqa: F401
from src.ai_agent.llm_providers import resolve_llm_config


def _resolve_chat_llm(ai_provider: Optional[str], custom_model: Optional[str], custom_base_url: Optional[str]) -> Dict[str, str]:
    """根据用户传参解析本次对话使用的 (model, base_url)（委托共享 SSOT）。"""
    return resolve_llm_config(ai_provider, custom_model, custom_base_url)


@router.post("/chat/send")
async def api_chat_send(req: ChatRequest):
    """聊天接口：POST /api/chat/send {session_id, message, choice?}
    → {kind, content, choices, tool_results, data_preview}
    """
    # choice 续接场景下 message 可为空（内容由下方拼接生成）
    if (not req.message or not req.message.strip()) and not req.choice:
        raise HTTPException(status_code=400, detail="消息不能为空")

    session = manager.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    # 诊断日志（定位"配置了 key 仍 AI 不可用"）：打印每次聊天请求前端到底传了什么
    print(
        f"[chat/send] session={req.session_id[:8]} provider={req.ai_provider!r} "
        f"api_key={((req.api_key or '')[:6] + '...') if req.api_key else 'None'} "
        f"model={req.custom_model!r} base_url={((req.custom_base_url or '')[:30] + '...') if req.custom_base_url else 'None'}"
    )
    if not session.datasets:
        raise HTTPException(status_code=404, detail="请先上传数据：当前会话没有可用数据集")

    # 上传即侦察：若未侦察过则补扫（确保 data_profile 已存）
    if not session.data_profile:
        try:
            from src.data_recon import scan
            df = manager.get_data(req.session_id)
            if df is not None:
                session.data_profile = scan(df)
        except Exception:
            pass

    # —— 多表支持（懒探测 + 自动合并）——
    # 当 session 存在 ≥2 张「非合并」数据集、且尚未生成过合并宽表时：
    #   1) 后台静默调用 build_analysis_units 识别关联键并合并；
    #   2) 把合并宽表注册成 is_merged 数据集（不抢占当前 active 视图）；
    # 这样用户上传两表后，下拉里会自动多出「合并宽表」这一选项（Q1=B 行为）。
    # 合并失败/无关联键则静默跳过，不影响正常单表对话。
    try:
        from backend.services.multi_table import maybe_auto_merge
        maybe_auto_merge(manager, req.session_id)
    except Exception as e:
        print(f"[chat/send] 多表自动合并探测被跳过：{type(e).__name__}: {e}")

    try:
        # 按请求里的模型配置实例化 agent；任一字段为空时落回默认值（Agnes）
        # —— 这样既支持用户自由切模型，又保留「未配置也能跑」的后备体验。
        provider_model = _resolve_chat_llm(
            ai_provider=req.ai_provider,
            custom_model=req.custom_model,
            custom_base_url=req.custom_base_url,
        )
        if req.api_key:
            agent = DataAnalysisAgent(
                api_key=req.api_key,
                model=provider_model["model"],
                base_url=provider_model["base_url"],
            )
        else:
            agent = DataAnalysisAgent(
                model=provider_model["model"],
                base_url=provider_model["base_url"],
            )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 把本次请求携带的 AI 配置持久化到会话：报告等「工具内部 LLM 调用」需要从会话
    # 读取用户所选 provider/key，否则会回退默认 Agnes 的 key → 401 → 报告降级为「纯汇总」。
    # 仅在用户本次带了新配置时写入，避免把会话里已有配置覆盖成空。
    try:
        if req.api_key or req.ai_provider:
            manager.set_api_config(
                req.session_id,
                req.api_key or (getattr(session, "api_key", "") or ""),
                req.ai_provider or (getattr(session, "ai_provider", "") or ""),
                req.custom_model or (getattr(session, "custom_model", "") or ""),
                req.custom_base_url or (getattr(session, "custom_base_url", "") or ""),
            )
    except Exception as e:
        print(f"[chat/send] 持久化 AI 配置失败（不影响本轮对话）：{type(e).__name__}: {e}")

    # 多轮：若用户点了 choice，把选择拼进消息；并恢复历史
    # 防御：只有属于 LEGAL_METHODS 的合法 method 才走"选择续接 + 执行清洗"分支；
    # 其余（含 [object Object] 等垃圾字符串）一律当普通消息处理，避免误执行清洗。
    choice = req.choice if (isinstance(req.choice, str) and req.choice in LEGAL_METHODS) else None
    message = req.message or '分析'
    if choice:
        message = f"我选择执行：{choice}（请调用 clean_data 工具执行该清洗方法）"
    history = session.messages if session.messages else None

    try:
        # agentic_chat 是同步的多轮 LLM 循环（可达数十秒），必须放入线程池执行；
        # 直接在 async def 里调用会阻塞事件循环，导致健康检查超时（Render 报警）
        # 与所有并发请求无响应（前端 Network Error）。与 upload 路由的 to_thread 策略一致。
        result = await asyncio.to_thread(
            agent.agentic_chat, message, req.session_id, history=history
        )
    except Exception as e:
        print("[chat/send] EXCEPTION traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI 调用失败：{str(e)}")

    # 写回对话历史，供下一轮续接。
    # 关键：剥离本轮内部的 tool_calls 与 role:"tool" 回灌消息，只保留
    # system/user/assistant 纯文字对话。否则下一轮 LLM 会把上一轮未完成的
    # 工具链误判为"待办任务"而反复重调工具，导致撑满 max_rounds 被强制截断。
    session.messages = _sanitize_history(result.get("messages", []))

    # 同步把这一轮的 user + 最终 assistant 回复落到持久化 chat_history。
    # 【历史记录要按这一刻】用户问了一句 → 必须有 user + assistant 两条落库，
    # 这样退出登录再登录后，历史回填能看到这个会话。
    #
    # 关键点：agentic_chat 走的是异步回调返回，result 里通常没有 messages 数组，
    # 最终 assistant 内容由 result["content"] / result.get("answer") 提供。
    # 这里多源兜底 + 把最终回复同步塞回 session.messages 以驱动后续多轮续接。
    assistant_content = (
        result.get("content")
        or result.get("answer")
        or ""
    )
    # 【修 bug】第 171 行已经把本轮 user 写进了 session.messages（agentic_chat 内部
    # 把 user 消息 append 进 messages，sanitize 后仍保留）。这里【不能再 append user】，
    # 否则会重复成两个 user 气泡。
    # 同时清理 sanitize 后残留的「空 assistant」——LLM 调工具那一轮的 assistant
    # content 为空，sanitize 只去掉了 tool_calls，会留下一条空白的 assistant 气泡。
    # 最终 assistant 文字（尤其 await_choice 场景的"发现 N 列缺失值…"）在
    # result["content"] 里、不在 messages 的 assistant 中，所以这里只补它。
    session.messages = [
        m for m in session.messages
        if not (m.get("role") == "assistant" and not (m.get("content") or "").strip())
    ]
    if assistant_content:
        # 去重：若 sanitize 后最后一条已是相同 content 的 assistant（LLM 调工具前
        # 自己写了文字，如"发现 N 列缺失值…"），就不再重复 append；只有清理掉
        # 空 assistant、或 messages 末尾不是 assistant 时才补最终文字。
        _last = session.messages[-1] if session.messages else None
        if not _last or _last.get("role") != "assistant" or _last.get("content") != assistant_content:
            session.messages.append({"role": "assistant", "content": assistant_content})

    # 历史会话记录的硬要求：用户发了问 → 必须有记录。
    # append_history 内部已 try/except 锁住，不会让请求失败。
    if req.message and req.message.strip():
        manager.append_history(req.session_id, "user", req.message)
    if assistant_content:
        manager.append_history(req.session_id, "assistant", assistant_content)

    # 若清洗后返回了数据预览占位，回填 head（取最新 merged/active df 前 5 行）
    data_preview = result.get("data_preview")
    if data_preview:
        try:
            df = manager.get_data(req.session_id)
            # 优先取 merged 宽表
            for did, ds in session.datasets.items():
                if getattr(ds, "is_merged", False):
                    mdf = manager.get_dataset_df(req.session_id, did)
                    if mdf is not None:
                        df = mdf
                        break
            if df is not None:
                data_preview["head"] = df.head(5).replace({float("nan"): None}).to_dict(orient="records")
        except Exception:
            pass

    return sanitize_json({
        "success": True,
        "kind": result.get("kind", "text"),
        "content": result.get("content", ""),
        "choices": result.get("choices", []),
        "tool_results": result.get("tool_results", []),
        "data_preview": data_preview,
    })


@router.get("/chat/messages")
async def api_chat_messages(session_id: str):
    """返回该会话的聊天历史（纯文字 user/assistant 对话流）。

    用于「历史会话」恢复后，智能对话页进入时回填上一次对话记录，
    让用户登录回来能看到「卖的最好的商品是什么 → 大模型回答」这类结果。
    """
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    # 优先用 session.messages：每次 /chat/send 时经 _sanitize_history 净化，
    # 仅含 system/user/assistant（已去 tool_calls）的纯文字对话。
    # 但 session.messages 不持久化——后端重启后内存丢了，只能从 DB 还原其他字段。
    # 此时退回到 session.chat_history（持久化字段，每条形如 {role, content, ts}），
    # 它的 role 只有 user/assistant，正好回填历史会话。
    msgs = session.messages or []
    # 过滤系统内部消息（打 _system_injected 标记的系统注入提示，如"数据已清洗完成
    # 请立即调用三分析工具"）与 LLM 调工具前的开场白（_preamble，如 "I'll run the
    # three analysis tools now."）。它们只服务 LLM 上下文，不该渲染成用户/助手气泡。
    # session.messages 本体保留这些消息：_clean_prompt_in_ctx 门禁判断依赖注入提示存在。
    msgs = [
        m for m in msgs
        if m.get("role") in ("user", "assistant")
        and not m.get("_system_injected")
        and not m.get("_preamble")
    ]
    source = "messages"
    if not msgs:
        ch = session.chat_history or []
        msgs = [{"role": h.get("role"), "content": h.get("content", "")}
                for h in ch if h.get("role") in ("user", "assistant")]
        source = "chat_history"
    # 挂起的清洗选择：上一轮弹了填充方式选项框但用户还没选。
    # 历史流是纯文字的，choices 不会随消息持久化，导致刷新后只剩一句
    # 「发现 N 列缺失值」却没有按钮可点。这里把挂起状态补成一条带 choices 的
    # assistant 消息，前端即可原样渲染出选择按钮。
    try:
        pending = (session.data_profile or {}).get("pending_clean")
    except Exception:
        pending = None
    if pending and pending.get("choices"):
        msgs = list(msgs) + [{
            "role": "assistant",
            "content": pending.get("summary") or "请选择缺失值填充方式：",
            "choices": pending.get("choices"),
        }]

    return sanitize_json({
        "success": True,
        "session_id": session_id,
        "messages": msgs,
        "source": source,
    })
