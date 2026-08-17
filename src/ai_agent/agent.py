"""
AI Agent 核心 - 使用原生 OpenAI 兼容 API 实现函数调用
不依赖 LangChain，更简单、更可控

默认 AI 提供方为 Agnes（OpenAI 兼容协议）。API Key 从环境变量 AGNES_API_KEY 读取。
"""
import os
import ast
import copy
import logging
import pandas as pd
import json
import openai
import threading
import unicodedata
import difflib
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("agent")

# 图表意图关键词：仅当用户消息含这些词时，才对 LLM 开放 generate_chart（延后到独立一轮）
_CHART_INTENT_KEYWORDS = (
    "图", "图表", "可视化", "画", "绘制", "柱状", "条形", "折线", "曲线", "饼图",
    "散点", "雷达", "热力", "树图", "趋势图", "分布图", "bar", "line", "pie", "chart",
    "plot", "可视化展示", "做个图", "来张图",
)

# 清洗意图关键词（收窄版）：仅含「明确动手/分析」词时，才对 LLM 开放
# clean_data 体检态（弹清洗选择框）。已删除「看看/咋样/最高/分布/对比」等口语词，
# 避免用户随口一问就被误判为要清洗——清洗必须经用户明确表达才触发。
_CLEAN_INTENT_KEYWORDS = (
    "分析", "跑模型", "清洗", "缺失", "缺失值", "预处理", "处理缺失", "去重",
    "修复", "整理数据", "清理",
)


# 产出工具意图关键词：报告/大屏仅在用户明确表达对应意图时，才对 LLM 开放对应产出工具。
_REPORT_INTENT_KEYWORDS = (
    "报告", "分析报告", "分析报表", "生成报告", "写报告", "出报告", "报告分析", "总结报告",
)
_BIGSCREEN_INTENT_KEYWORDS = (
    "大屏", "数据大屏", "可视化大屏", "驾驶舱", "看板大屏", "大屏展示", "大屏可视化",
)
# 分析意图关键词：已清洗后，用户表达"分析/跑模型/趋势/排行"等意愿才放开三分析工具，
# 避免用户随口聊天也被误放开动手工具。
_ANALYSIS_INTENT_KEYWORDS = (
    "分析", "跑模型", "业务模型", "趋势", "排行", "排名", "结构", "占比", "相关性",
    "异常", "分布", "统计", "洞察", "总结一下", "分析一下", "帮我分析",
)


def _has_chart_intent(text: str) -> bool:
    """判断用户消息是否包含图表/可视化意图。"""
    if not text:
        return False
    t = text.lower()
    return any(kw in t for kw in _CHART_INTENT_KEYWORDS)


def _has_report_intent(text: str) -> bool:
    """判断用户消息是否包含生成报告意图。"""
    if not text:
        return False
    t = text.lower()
    return any(kw in t for kw in _REPORT_INTENT_KEYWORDS)


def _has_bigscreen_intent(text: str) -> bool:
    """判断用户消息是否包含生成大屏意图。"""
    if not text:
        return False
    t = text.lower()
    return any(kw in t for kw in _BIGSCREEN_INTENT_KEYWORDS)


def _has_analysis_intent(text: str) -> bool:
    """判断用户消息是否包含调用三分析工具（业务模型/通用统计/自由写码）的意图。"""
    if not text:
        return False
    t = text.lower()
    return any(kw in t for kw in _ANALYSIS_INTENT_KEYWORDS)


def _has_clean_intent(text: str) -> bool:
    """判断用户消息是否包含明确的清洗/分析动手意图。

    命中才对 LLM 开放 clean_data（体检态，弹清洗选择框）；否则
    未清洗数据下也不放任何动手工具，杜绝 LLM 自作主张弹清洗框。
    """
    if not text:
        return False
    t = text.lower()
    return any(kw in t for kw in _CLEAN_INTENT_KEYWORDS)


def _parse_clean_method_from_message(text: str) -> Optional[str]:
    """从用户「我选择执行：xxx」消息中解析出清洗 method。

    匹配顺序：method 枚举字面量（fill_mean 等）优先，其次中文 label（均值填充等）。
    命中返回 method 字符串，未命中返回 None（交给 LLM 自行判断，不强制）。
    """
    if not text or "我选择执行" not in text:
        return None
    # method 字面量（含下划线写法）
    for m in ("fill_mean", "fill_median", "fill_mode", "fill_0"):
        if m in text:
            return m
    # 中文 label 映射
    label_map = {
        "均值填充": "fill_mean",
        "平均数填充": "fill_mean",
        "中位数填充": "fill_median",
        "众数填充": "fill_mode",
        "填0": "fill_0",
        "填零": "fill_0",
        "填充0": "fill_0",
    }
    for label, method in label_map.items():
        if label in text:
            return method
    return None

from src.ai_agent.prompts import (
    SYSTEM_PROMPT, REPORT_SYSTEM_PROMPT, REPORT_USER_PROMPT_TEMPLATE,
    INSIGHTS_SYSTEM_PROMPT, INSIGHTS_USER_PROMPT_TEMPLATE,
    REPORT_BI_SYSTEM_PROMPT, REPORT_BI_USER_PROMPT_TEMPLATE,
    CHAT_SYSTEM_PROMPT,
)
from src.report_analyzer import run_full_analysis
from src.report_builder import ReportBuilder
from src.report_builder import SECTION_DISPLAY_NAME

class DataAnalysisAgent:
    """数据分析 AI Agent（原生 DeepSeek API 实现）"""

    def __init__(self, api_key: str = None, model: str = "agnes-2.0-flash", base_url: str = "https://apihub.agnes-ai.com/v1"):
        """初始化 Agent（默认使用 Agnes，OpenAI 兼容协议）

        api_key 不传时从环境变量 AGNES_API_KEY 读取（后端 backend/.env 已内置）；
        若仍为空则显式报错，避免静默拿到 None 调 API。
        """
        if api_key is None:
            api_key = os.environ.get("AGNES_API_KEY", "")
        if not api_key:
            raise ValueError("缺少 AGNES_API_KEY：请在 backend/.env 中配置 Agnes API Key")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

        # 初始化 OpenAI 客户端
        # 报告生成最长 180s；openai SDK 默认 max_retries=2，一旦 AI 服务慢/不可达，
        # 单次超时(180s)后会再重试 2 次，最坏 180×3=540s，远超前端 300s 超时 →
        # 前端先 ECONNABORTED 断开，后端还在重试，用户永远收不到降级报告。
        # 故关闭 SDK 重试（max_retries=0）：超时即抛错 → 立即走 fallback 降级报告，
        # 保证后端在 180s 内返回 200，前端不会再超时。
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=240.0,
            max_retries=0,
        )

    def _get_data_summary(self, df: pd.DataFrame) -> str:
        """生成数据摘要（用于传给 AI）"""
        numeric_stats = ""
        try:
            if len(df.select_dtypes(include=['number']).columns) > 0:
                import io as _io
                buf = _io.StringIO()
                df.describe().to_csv(buf, encoding='utf-8')
                numeric_stats = buf.getvalue()
        except Exception:
            numeric_stats = "(无法计算描述性统计)"

        summary = f"""
数据规模：{len(df)} 行 x {len(df.columns)} 列
列名：{list(df.columns)}
数据类型：{dict(zip(df.columns, [str(dtype) for dtype in df.dtypes]))}
缺失值：{df.isnull().sum().to_dict()}
数值列统计：
{numeric_stats}
"""
        return summary

    def _execute_code(self, code: str, df: pd.DataFrame, timeout_sec: int = 18) -> str:
        """执行 Python 代码分析数据（带超时保护），返回文本结论字符串。

        兼容 analyze() 旧路径（第189行），内部委托模块级 _execute_code_structured，
        只取其中的 text 部分。结构化 chart 数据对旧路径无意义，丢弃。
        """
        return _execute_code_structured(code, df, timeout_sec).get("text", "")

    def analyze(self, user_query: str, df: pd.DataFrame, mode: str = "analysis") -> str:
        """分析用户问题，返回 AI 回答

        mode="analysis"（默认）：兼容旧逻辑——分析类请求走 generate_insights 返回结构化 JSON，
            通用对话走 SYSTEM_PROMPT 且保留代码执行能力（供分析页等旧调用方使用）。
        mode="chat"：纯对话模式——始终用 CHAT_SYSTEM_PROMPT，禁止代码/工具调用格式，
            不解析执行任何代码，直接把 AI 文本返回前端（聊天页用）。

        当用户询问分析/图表时，返回结构化 JSON（同 generate_insights 格式），
        包含 insights 和 intents，以便前端生成可执行的分析计划。
        """
        try:
            data_summary = self._get_data_summary(df)

            # ===== 聊天模式：纯对话，不走分析分支、不执行代码 =====
            if mode == "chat":
                messages = [
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"用户问题：{user_query}\n\n当前数据摘要：\n{data_summary}\n\n请直接用中文回答用户问题。"}
                ]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                    timeout=60,
                )
                return response.choices[0].message.content

            # ===== 分析模式（默认，保持旧行为）=====
            # 判断是否为分析/图表相关请求（覆盖自然语言变体）
            is_analysis_request = any(kw in user_query for kw in
                ['图表', '建议', '推荐', '分析方向', '做什么', '画什么', '地图', '省份', '词云', '词频',
                 '分析', '可视化', '生成', '统计', '对比', '趋势', '分布',
                 '规律', '特点', '特征', '关系', '变化', '增长', '下降', '排名',
                 '占比', '构成', '分类', '分组', '比较', '看看', '查看', '展示',
                 '画图', '作图', '怎么样', '如何', '哪个', '哪些'])

            if is_analysis_request:
                # 分析请求：直接调用 generate_insights 返回结构化 JSON
                result = self.generate_insights(df, user_query)
                # V3 意图补强已移除（analysis_library 删除，新流程由列名匹配引擎决定分析）
                return result

            # 通用对话：直接回答
            _chart_hint = (
                '\n\n如果用户询问分析方向或图表建议，请在\u201c分析建议\u201d章节中，'
                '每条建议包含 (X:列名, Y:列名) 格式标注，并紧跟\u201c\u2192 图表类型\u201d说明。'
                '示例格式：\n'
                "1. 各省份销售金额的地区分布 → 3D地图（X:省份, Y:销售金额）\n"
                "   + 汇总表格（行:省份, 列:销售金额）\n"
                "2. 各产品类别的销售对比 → 柱状图（X:产品类别, Y:销售金额）\n"
                "   + 排序表格（排序:销售金额, 降序）\n"
                "请使用数据中真实的列名，不要虚构不存在的列。"
            ) if any(kw in user_query for kw in ['图表', '建议', '推荐', '分析方向', '做什么', '画什么', '地图', '省份']) else ""
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"用户问题：{user_query}\n\n当前数据摘要：\n{data_summary}\n\n请直接用中文回答用户问题。如果需要计算，在回答中说明分析思路即可，不要生成 Python 代码。{_chart_hint}"}
            ]

            # 调用 API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
                timeout=60,
            )

            ai_response = response.choices[0].message.content

            # 仅在确实需要执行代码时才执行（带超时保护）
            if "```python" in ai_response or "```py" in ai_response:
                code_start = ai_response.find("```python")
                if code_start == -1:
                    code_start = ai_response.find("```py")

                code_end = ai_response.find("```", code_start + 10)
                if code_start != -1 and code_end != -1:
                    code = ai_response[code_start:code_end].replace("```python", "").replace("```py", "").replace("```", "").strip()

                    # 执行代码（20 秒超时）
                    execution_result = self._execute_code(code, df)

                    follow_up_messages = messages + [
                        {"role": "assistant", "content": ai_response},
                        {"role": "user", "content": f"代码执行结果：\n{execution_result}\n\n请根据这个结果，用中文总结分析结论。"}
                    ]

                    follow_up_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=follow_up_messages,
                        temperature=0.3,
                        max_tokens=2048,
                        timeout=60,
                    )

                    return follow_up_response.choices[0].message.content

            return ai_response

        except Exception as e:
            return f"AI 分析出错：{str(e)}\n\n请检查 API Key 是否正确，或稍后重试。"

    def generate_insights(self, df: pd.DataFrame, user_query: str = "") -> str:
        """自动生成数据洞察报告 + 分析意图列表（JSON 格式）
        
        使用 INSIGHTS_SYSTEM_PROMPT + INSIGHTS_USER_PROMPT_TEMPLATE，
        输出 JSON：{insights: Markdown, intents: [{business_question, analysis_goal, priority, reason}]}
        
        参数：
            user_query: 用户的具体分析问题（可选），如果提供，会在提示词中加入该问题
        """
        try:
            # ============================================
            # 阶段 1-3：Python 精确统计分析
            # ============================================
            analysis_data = run_full_analysis(df, None)
            fields = analysis_data["phase_1_fields"]
            stats = analysis_data["phase_3_stats"]
            charts = analysis_data["phase_2_charts"]

            # ---- 构建数据摘要（供 LLM 使用）----
            data_summary = _build_insights_data_summary(df, fields, stats, charts)

            # ============================================
            # 阶段 4-5：AI 生成洞察（Structured Output JSON）
            # ============================================
            query_context = f"\n\n用户具体问题：{user_query}\n请重点围绕用户问题生成相关的分析意图。" if user_query else ""
            user_prompt = INSIGHTS_USER_PROMPT_TEMPLATE.format(
                data_summary=data_summary,
            ) + query_context

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": INSIGHTS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                    timeout=120,
                )

                ai_text = response.choices[0].message.content or ""
                return ai_text  # 返回 JSON 字符串，由 insights.py 解析

            except Exception as e:
                # AI 调用失败时，降级为纯统计洞察
                import json as _json
                fallback = _build_fallback_insights(df, fields, stats, charts, str(e))
                return _json.dumps({"insights": fallback, "intents": []})

        except Exception as e:
            import json as _json
            return _json.dumps({"insights": f"生成洞察报告出错：{str(e)}", "intents": []})

    # ======================================================================
    # Chat 智能体：OpenAI function calling 循环
    # ======================================================================

    def _get_llm_cfg(self) -> Dict[str, Any]:
        """构造列名映射/合并需要的 llm_cfg（与 analysis.py 一致）。"""
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
        }

    def _current_df(self, manager, session_id: str):
        """取当前参与分析的 df：
        - 优先取 is_merged 宽表（清洗后入库的），否则 active_df。
        - 单表直接用；多表由 _resolve_tool_call 的 clean_data 分支负责合并。
        """
        session = manager.get_session(session_id)
        if session is None:
            return None
        # 优先 merged 数据集
        for did, ds in session.datasets.items():
            if getattr(ds, "is_merged", False):
                df = manager.get_dataset_df(session_id, did)
                if df is not None:
                    return df
        # 否则 active
        if session.active_dataset_id:
            return manager.get_dataset_df(session_id, session.active_dataset_id)
        return None

    def _merge_and_register(self, manager, session_id: str, llm_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """把 session 里所有非合并数据集按关联键合并，并注册成 is_merged 宽表。

        返回 {status, dataset_id?, summary, data}；无关联键/合并失败则 status='skip'，
        不抛异常。供 _resolve_tool_call 的 merge_tables 分支与后端懒合并复用同一套逻辑。
        """
        session = manager.get_session(session_id)
        if session is None:
            return {"tool": "merge_tables", "status": "fail", "summary": "会话不存在", "data": {}}
        non_merged = [
            did for did, ds in session.datasets.items()
            if not getattr(ds, "is_merged", False)
        ]
        pairs = [(did, manager.get_dataset_df(session_id, did)) for did in non_merged]
        valid = [(did, df) for did, df in pairs if df is not None and not df.empty]
        if len(valid) < 2:
            return {"tool": "merge_tables", "status": "skip",
                    "summary": "当前只有一个可用数据集，无需合并。", "data": {}}

        from src.merge.dataset_merger import build_analysis_units
        file_names = {did: (session.datasets[did].file_name or did) for did, _ in valid}
        units = build_analysis_units(valid, file_names=file_names, llm_cfg=llm_cfg)
        merged_unit = next((u for u in units if u.kind == "merged"), None)
        if merged_unit is None or merged_unit.df is None:
            return {"tool": "merge_tables", "status": "skip",
                    "summary": "未检测到可关联的字段，无法自动合并（可改为手动指定关联键）。",
                    "data": {}}

        did = manager.add_merged_dataset(
            session_id, merged_unit.df,
            sources=merged_unit.sources, keys=merged_unit.keys,
            file_name="合并宽表",
        )
        return {
            "tool": "merge_tables", "status": "ok",
            "summary": f"已按关联键 {merged_unit.keys} 合并 {len(merged_unit.sources)} 张表，"
                       f"生成宽表（{merged_unit.df.shape[0]} 行 × {merged_unit.df.shape[1]} 列）。",
            "data": {"dataset_id": did, "rows": int(merged_unit.df.shape[0]),
                     "columns": list(merged_unit.df.columns), "keys": merged_unit.keys},
        }

    def _resolve_tool_call(self, name: str, args: Dict[str, Any], manager, session_id: str,
                           user_message: str = "") -> Dict[str, Any]:
        """执行单个工具调用，返回 {tool, status, summary, data}。

        df 由这里注入（不依赖 tools_registry.get_tool，避免缺 df 报错）。
        user_message 用于判断 clean_data 的 method 是否来自用户 choice 续接（防 LLM 跳过弹窗）。
        """
        llm_cfg = self._get_llm_cfg()
        # 注：数据侦察（profile_data）在分析对话中不再对 LLM 开放——
        # 上传时后端已自动侦察并存入 session.data_profile，且数据快照由
        # _snapshot_for_prompt 注入上下文，LLM 不需要也不能再调用侦察工具。

        if name == "merge_tables":
            # 多表自动合并：识别关联键链式 join，生成 is_merged 宽表（不抢占 active）。
            # 合并后用户可在前端下拉里看到「合并宽表」这一选项，自由切换分析。
            llm_cfg = self._get_llm_cfg()
            return self._merge_and_register(manager, session_id, llm_cfg)

        if name == "clean_data":
            # 先判断单表/多表
            session = manager.get_session(session_id)
            if session is None:
                return {"tool": name, "status": "fail", "summary": "会话不存在", "data": {}}
            datasets = [(did, manager.get_dataset_df(session_id, did))
                        for did in session.datasets.keys()]
            valid = [(did, df) for did, df in datasets if df is not None and not df.empty]

            method = args.get("method")
            # 门禁（修 bug，非强制拦截）：clean_data 的"执行态"只能来自用户 choice 续接。
            # 若 LLM 在第一轮就自行带 method 调用（未经用户从弹窗选择），视为跳过体检态，
            # 忽略 method、强制退回体检态，让系统先把可选项弹给用户。
            is_choice_continuation = "我选择执行" in (user_message or "")
            print(f"[DEBUG 门禁] method={method!r} type={type(method).__name__} | is_choice_continuation={is_choice_continuation} | user_message={user_message!r}")
            if method and not is_choice_continuation:
                logger.warning("clean_data 门禁：LLM 未经用户选择就带 method=%s 调用，强制退回体检态", method)
                method = None
            print(f"[DEBUG 门禁] after method={method!r}")

            # 体检态（无 method）→ 直接对当前 df 扫描建议
            if not method:
                from tools_registry import clean_data as _clean_data
                cur = self._current_df(manager, session_id)
                res = _clean_data(cur, None)
                return {"tool": name, "status": "ok" if res.ok else "fail",
                        "summary": res.message or "",
                        "data": res.data if res.ok else {"error": res.error},
                        "await_choice": True}

            # 执行态：单表直接取，多表先合并
            if len(valid) <= 1:
                merged_df = self._current_df(manager, session_id)
                sources = [did for did, _ in valid]
                keys: List[str] = []
            else:
                from src.merge.dataset_merger import build_analysis_units
                file_names = {did: (session.datasets[did].file_name or did) for did, _ in valid}
                units = build_analysis_units(valid, file_names=file_names, llm_cfg=llm_cfg)
                merged_unit = next((u for u in units if u.kind == "merged"), None)
                if merged_unit is None or merged_unit.df is None:
                    # 无关联键/无法合并 → 退化为所有表纵向拼接，保证可执行
                    merged_df = pd.concat([df for _, df in valid], ignore_index=True, sort=False)
                    sources = [did for did, _ in valid]
                    keys = []
                else:
                    merged_df = merged_unit.df
                    sources = merged_unit.sources
                    keys = merged_unit.keys

            # 列名映射（跟 analysis.py 流水线一致）
            from src.mapping.column_mapper import map_dataset_columns
            try:
                mapped_df = map_dataset_columns(session_id, None, merged_df, llm_cfg)
            except Exception as e:
                mapped_df = merged_df  # 映射失败降级为不映射，避免阻断

            # 执行清洗
            from tools_registry import clean_data as _clean_data
            actions = [{"method": method}]
            res = _clean_data(mapped_df, actions)
            if not res.ok:
                return {"tool": name, "status": "fail", "summary": res.error or "清洗失败", "data": {}}
            cleaned_df = res.data.get("cleaned_df")
            summary = res.data.get("summary", {})

            # 写回 session：注册成 merged 宽表（不抢占 active 视图）
            if cleaned_df is not None:
                manager.add_merged_dataset(
                    session_id, cleaned_df, sources=sources, keys=keys,
                    file_name="聊天清洗宽表",
                )
            preview_rows = int(cleaned_df.shape[0]) if cleaned_df is not None else 0
            preview_cols = list(cleaned_df.columns) if cleaned_df is not None else []

            # 清洗完成后仅返回清洗结果。不在此处代码硬串联三分析（方案X已废弃）：
            # 改法1（最小闭环）：清洗执行态成功后，agentic_chat 会在本轮 tool_results 检测
            # 到 clean_data=ok，置 _need_inject_clean_prompt 标志；下一轮循环顶部据此向
            # messages 注入一条"请立即调用三分析工具"的 user 提示（并立即清标志防死循环），
            # 由 LLM 自动调用 run_template / run_analysis / run_python，
            # 三个工具把完整 AnalysisPackage 写入 session.analysis_packages，供产出工具消费。

            return {
                "tool": name, "status": "ok",
                "summary": res.message or "清洗完成",
                # preview_rows / preview_cols 由 cleaned_df 直接推导，不依赖 clean_data 内部键名
                "data": {"summary": summary,
                         "preview_rows": preview_rows,
                         "preview_cols": preview_cols},
            }

        if name == "run_template":
            df = self._current_df(manager, session_id)
            from src.mapping.column_mapper import map_dataset_columns
            try:
                mapped_df = map_dataset_columns(session_id, None, df, llm_cfg)
            except Exception:
                mapped_df = df
            from tools_registry import run_template
            intents = args.get("intents") or []
            # 传入 manager / session_id：run_template 会把完整 AnalysisPackage 写入
            # session.analysis_packages，供 generate_chart / build_dashboard / generate_report 读取。
            res = run_template(mapped_df, intents, manager=manager, session_id=session_id)
            return {"tool": name, "status": "ok" if res.ok else "fail",
                    "summary": res.message if res.ok else (res.error or f"{name} 工具调用失败"),
                    "data": res.data if res.ok else {"error": res.error}}

        if name == "run_analysis":
            df = self._current_df(manager, session_id)
            from src.mapping.column_mapper import map_dataset_columns
            try:
                mapped_df = map_dataset_columns(session_id, None, df, llm_cfg)
            except Exception:
                mapped_df = df
            from tools_registry import run_analysis
            res = run_analysis(mapped_df)
            return {"tool": name, "status": "ok" if res.ok else "fail",
                    "summary": res.message if res.ok else (res.error or f"{name} 工具调用失败"),
                    "data": res.data if res.ok else {"error": res.error}}

        if name == "generate_chart":
            df = self._current_df(manager, session_id)
            from src.mapping.column_mapper import map_dataset_columns
            try:
                mapped_df = map_dataset_columns(session_id, None, df, llm_cfg)
            except Exception:
                mapped_df = df
            from tools_registry import generate_chart
            logger.info("generate_chart 入参: chart_type=%s args=%s", args.get("chart_type"), args)
            # 去掉 df 和 chart_type 后，其余参数透传给 create_chart
            chart_kwargs = {k: v for k, v in args.items() if k not in ("df", "chart_type")}
            res = generate_chart(mapped_df, args.get("chart_type", ""), **chart_kwargs)
            return {"tool": name, "status": "ok" if res.ok else "fail",
                    "summary": res.message if res.ok else (res.error or f"{name} 工具调用失败"),
                    "data": res.data if res.ok else {"error": res.error}}

        if name == "run_python":
            df = self._current_df(manager, session_id)
            from tools_registry import run_python
            logger.info("run_python 入参 code=\n%s", args.get("code", ""))
            res = run_python(df, args.get("code", ""))
            return {"tool": name, "status": "ok" if res.ok else "fail",
                    "summary": res.message if res.ok else (res.error or f"{name} 工具调用失败"),
                    "data": res.data if res.ok else {"error": res.error}}

        if name == "generate_report":
            from tools_registry import generate_report
            # 产出工具吃 session.analysis_packages（由三分析工具写入的完整 AnalysisPackage）。
            res = generate_report(manager, session_id)
            return {"tool": name, "status": "ok" if res.ok else "fail",
                    "summary": res.message if res.ok else (res.error or f"{name} 工具调用失败"),
                    "data": res.data if res.ok else {"error": res.error}}

        if name == "build_dashboard":
            from tools_registry import build_dashboard
            # 产出工具吃 session.analysis_packages（由三分析工具写入的完整 AnalysisPackage）。
            res = build_dashboard(manager, session_id)
            return {"tool": name, "status": "ok" if res.ok else "fail",
                    "summary": res.message if res.ok else (res.error or f"{name} 工具调用失败"),
                    "data": res.data if res.ok else {"error": res.error}}

        return {"tool": name, "status": "fail", "summary": f"未知工具：{name}", "data": {}}

    def _is_dataset_cleaned(self, manager, session_id: str) -> bool:
        """状态位：session 中是否已存在 is_merged 清洗宽表（clean_data 执行态写过）。

        上传时后端只写 session.data_profile、不建 is_merged 宽表，故初始为 False；
        清洗完成后置 True，主循环据此判定「已清洗」并决定是否放开 generate_chart。
        """
        try:
            session = manager.get_session(session_id)
            if session is None:
                return False
            return any(
                getattr(d, "is_merged", False)
                for d in session.datasets.values()
            )
        except Exception:
            return False

    def agentic_chat(self, message: str, session_id: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Chat 智能体主入口：function calling 循环 + 结构化响应。

        参数：
        - message：用户本轮消息
        - session_id：会话 ID（用于取 df / 写回清洗结果 / 存对话历史）
        - history：已有对话 messages（含 system/user/assistant/tool 角色），None 表示从零构建

        返回：
        {
            "kind": "text" | "choice" | "tool_executing",
            "content": str,
            "choices": [{"id","label","description"}, ...],   # kind="choice" 才有
            "tool_results": [{"tool","status","summary"}, ...],
            "data_preview": {"rows","columns","head"},        # 清洗后可选
            "messages": [...],                                 # 更新后的完整 messages，供路由写回 session
        }
        """
        from backend.services.session_manager import manager as _default_manager
        manager = _default_manager

        # 构建 messages
        if history is not None:
            messages = list(history)
            # 确保 system 在最前
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}] + messages
        else:
            messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
            # 附上数据快照，帮助 LLM 理解数据（不污染用户消息）
            snap = self._snapshot_for_prompt(manager, session_id)
            if snap:
                messages.append({"role": "system", "content": f"当前数据快照：\n{snap}"})

        messages.append({"role": "user", "content": message})

        # 确定性兜底（符合用户设计：用户选完清洗后必须接着执行并自动三分析，
        # 不能赌 LLM 自己调 clean_data 执行态——实测证明 LLM 会偷懒只回文字）。
        # 当本轮用户消息表达「我选择执行 xxx」且数据尚未清洗完成时，后端直接以
        # 执行态跑 clean_data（从用户消息里解析 method 关键词），不依赖 LLM 发 tool_calls。
        # 跑成功后置 _need_inject_clean_prompt，下一轮顶部注入提示驱动三分析。
        tool_results: List[Dict[str, Any]] = []
        # 改法1 标志：本轮若检测到 clean_data 执行态成功，置为 True；
        # 下一轮循环顶部据此注入"请立即调用三分析工具"提示后立即清为 False。
        # 作用域仅限本次 agentic_chat 调用，不持久化到 session，避免跨请求串状态。
        # ★ 必须在「确定性执行清洗」逻辑之前初始化，否则下方置 True 后又被此处
        #   （若误放在后面）的 =False 覆盖，导致三分析提示永远注入不了（历史 bug）。
        _need_inject_clean_prompt = False
        _choice_method = _parse_clean_method_from_message(message)
        if _choice_method and not self._is_dataset_cleaned(manager, session_id):
            print(f"[agentic_chat] 检测到用户选择清洗方式={_choice_method}，后端确定性执行 clean_data 执行态")
            _exec_res = self._resolve_tool_call(
                "clean_data", {"method": _choice_method}, manager, session_id,
                user_message=message,
            )
            tool_results.append({
                "tool": _exec_res.get("tool"),
                "status": _exec_res.get("status"),
                "summary": _exec_res.get("summary"),
                "await_choice": _exec_res.get("await_choice", False),
                "data": _exec_res.get("data", {}),
            })
            if _exec_res.get("tool") == "clean_data" and _exec_res.get("status") == "ok" \
                    and not _exec_res.get("await_choice"):
                _need_inject_clean_prompt = True
                print(f"[agentic_chat] 确定性执行 clean_data 成功 -> set _need_inject_clean_prompt")

        function_defs = None
        try:
            # 注意：tools_registry 位于 src/ 包下，需用 src. 前缀 import
            # （sys.path 含 project_root，src 是其下包；顶层 tools_registry.py 不存在，
            #  之前 from tools_registry 会抛 ModuleNotFoundError 被静默吞掉，导致 tools 永不传给 LLM）
            from src.tools_registry import get_function_definitions
            function_defs = get_function_definitions()
        except Exception as e:
            # 打出来，避免再被静默吞掉导致"LLM 不调工具"却查不到原因
            print(f"[agentic_chat] WARN get_function_definitions failed: {type(e).__name__}: {e}")
            function_defs = None

        max_rounds = 8
        # 本轮对话工具调用计数（函数内局部变量，每轮 agentic_chat 重建，不持久化到 session）。
        # 用于重复调用熔断：防止 LLM 在 function calling 循环里对同一工具反复调用、
        # 撑满 max_rounds 被强制截断（切断大类 A 死循环源：循环无重复调用上限）。
        # 只读快照类工具严格限 1 次；动手/出图类工具给少量冗余上限防失控。
        _tool_call_counts: Dict[str, int] = {}
        _TOOL_CALL_LIMITS: Dict[str, int] = {
            "profile_data": 1,
            "run_template": 3,
            "run_analysis": 3,
            "run_python": 3,
            "generate_chart": 3,
            "generate_report": 3,
            "build_dashboard": 3,
            "clean_data": 3,
        }
        # 注入提示文案（一次性，逼 LLM 主动发三分析 tool_calls）
        _CLEAN_DONE_PROMPT = (
            "数据已清洗完成。请立即依次调用以下三个分析工具对清洗后数据进行分析，"
            "不要再询问用户是否要分析："
            "①run_template（业务模型分析）"
            "②run_analysis（通用统计分析）"
            "③run_python（自由写码分析）。"
            "三者都调用完成后，写一段完整中文总结，必须覆盖每一个工具的结论。"
        )

        for _round in range(max_rounds):
            # 改法1：清洗完成后下一轮顶部注入提示，逼 LLM 自动调三分析工具。
            # 注入后立即清标志——即便本轮 LLM 未发 tool_calls 直接返回，下下轮也绝不再注入，彻底无死循环。
            if _need_inject_clean_prompt:
                messages.append({"role": "user", "content": _CLEAN_DONE_PROMPT})
                _need_inject_clean_prompt = False
                print(f"[agentic_chat] round={_round} injected clean-done prompt -> auto-run 3 analysis tools")
            _continue_count = 0
            _MAX_CONTINUE = 2
            _in_continue_phase = False
            try:
                create_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "timeout": 120,
                }
                if function_defs:
                    # 状态机下发（意图门禁版）：清洗是平等 LLM 工具，三分析/产出工具均由 LLM 按意图调用。
                    # ① 未清洗 + 用户原话含清洗意图词 → 只放开 clean_data（体检态弹选项框）
                    # ② 已清洗 + 分析词 → 放开三分析工具（run_template / run_analysis / run_python）
                    # ③ 已清洗 + 图表词 → 放开 generate_chart
                    # ④ 已清洗 + 报告词 → 放开 generate_report
                    # ⑤ 已清洗 + 大屏词 → 放开 build_dashboard
                    # ⑥ 其余（纯聊天/提问/无对应意图词）→ 不放任何动手工具，纯文字回答
                    #    注意：三个分析工具写入 session.analysis_packages，产出工具（图/大屏/报告）读它作为输入。
                    def _names_eq(t, n):
                        return t.get("function", {}).get("name") == n
                    # —— 续接总结阶段：上一轮已执行工具但 LLM 未给出完整总结。
                    # 本轮回灌一条"请总结"提示，且强制不放任何动手工具，确保 LLM 只写文字结论，
                    # 不再调工具导致再次中断（解决"分析到一半就断、需用户追问才继续"）。
                    if _in_continue_phase:
                        tools_for_round = []
                    else:
                        is_cleaned = self._is_dataset_cleaned(manager, session_id)
                        # 确定性清洗前置门禁（按用户设计）：
                        # 未清洗时，后端读数据侦察的缺失值结论来决定放行什么工具，
                        # 不允许 LLM 自主越级调三分析工具。
                        #   - 有缺失值 → 只放 clean_data，由 LLM 调体检态弹框供用户点选；
                        #   - 无缺失值 → 直接跳过清洗，放开三分析工具。
                        # 无论哪种，未清洗轮绝不放出图/大屏/报告工具。
                    if not is_cleaned:
                        _dp = getattr(manager.get_session(session_id), "data_profile", None) or {}
                        _overview = _dp.get("missing_overview") or {}
                        _total_missing = _overview.get("total_missing", 0) or 0
                        if _total_missing > 0:
                            tools_for_round = [t for t in function_defs if _names_eq(t, "clean_data")]
                        else:
                            # 无缺失：跳过清洗，直接进入三分析阶段
                            tools_for_round = [t for t in function_defs
                                               if _names_eq(t, "run_template")
                                               or _names_eq(t, "run_analysis")
                                               or _names_eq(t, "run_python")]
                    elif is_cleaned:
                        # 防御性放开：只要上下文里已存在「清洗完成→请调三分析」注入提示
                        # （该提示永久留在 messages 中，不像 _need_inject_clean_prompt 局部标志那样被清），
                        # 就强制放开三分析工具，保证 LLM 即便首轮 message 不含分析词也能收到工具列表。
                        # 注意：不能依赖 _need_inject_clean_prompt（顶部注入后已立即清为 False，本轮回看为 False），
                        # 只能靠 messages 中是否含注入提示来判定「已进入自动三分析阶段」。
                        _clean_prompt_in_ctx = any(
                            m.get("role") == "user" and _CLEAN_DONE_PROMPT in (m.get("content") or "")
                            for m in messages
                        )
                        # 顺序很重要：明确的用户意图（分析/图表/报告/大屏）必须先于兜底判定，
                        # 否则「刚清洗完」的 _clean_prompt_in_ctx 永久为 True 会把所有后续消息都吞成三分析，
                        # 导致生成大屏/报告/出图永远收不到工具（表现为点了没反应）。
                        if _has_analysis_intent(message):
                            tools_for_round = [t for t in function_defs
                                               if _names_eq(t, "run_template")
                                               or _names_eq(t, "run_analysis")
                                               or _names_eq(t, "run_python")]
                        elif _has_chart_intent(message):
                            tools_for_round = [t for t in function_defs if _names_eq(t, "generate_chart")]
                        elif _has_report_intent(message):
                            tools_for_round = [t for t in function_defs if _names_eq(t, "generate_report")]
                        elif _has_bigscreen_intent(message):
                            tools_for_round = [t for t in function_defs if _names_eq(t, "build_dashboard")]
                        elif _clean_prompt_in_ctx:
                            # 兜底：刚清洗完且用户本轮没说任何明确意图词 → 放开三分析（驱动自动补跑）。
                            # 仅作最后兜底，不抢占上面的明确意图。
                            tools_for_round = [t for t in function_defs
                                               if _names_eq(t, "run_template")
                                               or _names_eq(t, "run_analysis")
                                               or _names_eq(t, "run_python")]
                        else:
                            tools_for_round = []
                    else:
                        tools_for_round = []
                    # 没有任何工具可下放时，强制 LLM 直接文字回答（不允许它自己发明工具调用）
                    if not tools_for_round:
                        create_kwargs.pop("tools", None)
                        create_kwargs.pop("tool_choice", None)
                    else:
                        create_kwargs["tools"] = tools_for_round
                        create_kwargs["tool_choice"] = "auto"
                response = self.client.chat.completions.create(**create_kwargs)
            except Exception as e:
                return {
                    "kind": "text",
                    "content": f"AI 服务调用失败：{str(e)}",
                    "choices": [], "tool_results": tool_results,
                    "data_preview": None, "messages": messages,
                }

            msg = response.choices[0].message
            # 诊断日志（print 不受 logging 级别过滤，确保终端可见）
            print(f"[agentic_chat] round={_round} has_tool_calls={bool(msg.tool_calls)} content_len={len(msg.content or '')}")
            # 把 assistant 消息（含 tool_calls）加入历史
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    } for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            # 没有工具调用 → 可能是「完整最终回答」或「已执行工具但未总结完整」
            if not msg.tool_calls:
                content = msg.content or ""
                content = self._strip_tool_tags(content)  # 兜底：剥离可能泄露的工具标签

                # 续接机制：若本轮确实执行过工具（tool_results 非空），但 LLM 给出的 content
                # 明显不完整（没覆盖工具结论），则不直接收尾，而是补一轮强制总结。
                # 这样避免「分析跑到一半、LLM 主动收尾文字 → 循环结束 → 用户需再追问」的问题。
                ok_results = [tr for tr in tool_results if tr.get("status") == "ok"]
                # 兜底：若 LLM 再次返回空 content（即使在续接轮），不再追加 prompt 避免死循环；
                # 续接一轮后仍未拿到文字，跳出强制走 fallback 兜底，防止空白气泡。
                if _in_continue_phase and not content.strip():
                    _continue_count = _MAX_CONTINUE  # 锁住后续续接计数
                _needs_continue = (
                    not _in_continue_phase
                    and _continue_count < _MAX_CONTINUE
                    and bool(ok_results)
                    and content.strip()
                    and not self._is_complete_summary(content, ok_results)
                )
                if _needs_continue:
                    _continue_count += 1
                    _in_continue_phase = True
                    print(f"[agentic_chat] round={_round} continue-phase: 工具已执行但总结不完整，补一轮强制总结")
                    messages.append({
                        "role": "user",
                        "content": (
                            "你已通过工具完成分析。请现在用中文给出**完整**的总结，"
                            "必须覆盖每一个已执行工具的分析结论（不要只写建议、不要中断）。"
                            "不要再次调用任何工具，直接输出文字结论即可。"
                        ),
                    })
                    continue  # 进入下一轮：_in_continue_phase=True → 不放工具 → LLM 纯文字总结

                # 规范兜底：若 LLM 未写文字总结，但确实跑出了 ok 结果，
                # 补一轮请求强制 LLM 基于全部结果写完整总结（不污染 messages 历史）
                if not content.strip() and ok_results:
                    content = self._force_summary_from_results(messages, ok_results)
                    content = self._strip_tool_tags(content)

                # 最终兜底：避免把空 content 推给前端造成"空气泡"——若经过一切补刀后
                # 仍为空，根据是否有 ok 工具结果给用户一句可见的中性说明，让前端能渲染
                # 出文字，而不是空白消息。常见触发：模型连续多次返回空 content（超时/截断）。
                if not content.strip():
                    if ok_results:
                        content = "分析已完成，详情见下方图表与执行过程。"
                    else:
                        content = "未能生成回复，请稍后重试或换一个问题。"
                    print(f"[agentic_chat] round={_round} fallback: LLM 返回空 content，写入兜底文字")
                # 判断是否是在等用户选择（上一轮有 await_choice 工具）
                return {
                    "kind": self._classify_response(content, tool_results),
                    "content": content,
                    "choices": self._extract_choices(tool_results),
                    "tool_results": tool_results,
                    "data_preview": self._build_data_preview(tool_results),
                    "messages": messages,
                }

            # 执行每个 tool_call
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                # —— 清洗前置硬卡 ——
                # 仅当「未清洗 且 数据有缺失值」时，LLM 不得越级调用三分析/产出工具，
                # 必须走 clean_data 弹框流程（用户点选后才洗）。
                # 若「未清洗 但 无缺失值」，属于用户定义的「跳过清洗直接三分析」合法路径，
                # 不拦截（门禁已在上一轮把三分析工具下放给 LLM）。
                # 这是确定性门禁的兜底，根治「未清洗有缺失却直接分析」的日志 bug。
                if not self._is_dataset_cleaned(manager, session_id):
                    _dp = getattr(manager.get_session(session_id), "data_profile", None) or {}
                    _ov = _dp.get("missing_overview") or {}
                    _tm = _ov.get("total_missing", 0) or 0
                    if _tm > 0:
                        _ANALYSIS_TOOLS = {"run_template", "run_analysis", "run_python"}
                        _OUTPUT_TOOLS = {"generate_chart", "generate_report", "build_dashboard"}
                        if tool_name in _ANALYSIS_TOOLS or tool_name in _OUTPUT_TOOLS:
                            print(f"[agentic_chat] BLOCK越级 tool={tool_name}（未清洗且有缺失，按门禁驳回）")
                            tool_results.append({
                                "tool": tool_name,
                                "status": "ok",
                                "summary": "数据尚未清洗（且存在缺失值），按既定流程须先完成清洗才能分析/出图。"
                                           "请先调用 clean_data 进入清洗流程（或直接基于已有快照给出说明）。",
                                "await_choice": False,
                                "data": {},
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": "该操作被门禁拦截：数据尚未清洗且有缺失值。请先调用 clean_data 完成清洗流程。",
                        })
                        continue
                try:
                    fn_args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    fn_args = {}
                print(f"[agentic_chat] round={_round} tool={tool_name} args={fn_args}")

                # —— 重复调用熔断 ——
                # 同一轮对话内该工具已被调用达到上限时，不再执行真工具，直接返回
                # 固定提示逼 LLM 收口（不再回灌完整 data、不 append 新 tool 消息）。
                _tool_call_counts[tool_name] = _tool_call_counts.get(tool_name, 0) + 1
                _limit = _TOOL_CALL_LIMITS.get(tool_name, 3)
                if _tool_call_counts[tool_name] > _limit:
                    print(f"[agentic_chat] circuit-break tool={tool_name} count={_tool_call_counts[tool_name]} limit={_limit}")
                    tool_results.append({
                        "tool": tool_name,
                        "status": "ok",
                        "summary": "数据快照已在上下文中，无需重复调用；请基于已有信息直接给出分析结论或调用其他工具。",
                        "await_choice": False,
                        "data": {},
                    })
                    # 把熔断提示作为 tool 消息回灌，让 LLM 看到"该工具已饱和"，促其收口
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "该工具已达到调用上限，数据快照已在上下文中，请直接给出结论。",
                    })
                    continue

                result = self._resolve_tool_call(tool_name, fn_args, manager, session_id,
                                                 user_message=message)
                print(f"[agentic_chat] round={_round} tool_result status={result.get('status')} summary={result.get('summary','')[:200]} await_choice={result.get('await_choice')}")

                # 体检态：暂停等用户选择，不回灌 LLM、不继续循环
                if result.get("await_choice"):
                    tool_results.append({
                        "tool": result.get("tool"),
                        "status": result.get("status"),
                        "summary": result.get("summary"),
                        "await_choice": True,
                        "data": result.get("data", {}),
                    })
                    content = self._strip_tool_tags(
                        result.get("summary") or "已扫描数据，请选择缺失值填充方式："
                    )
                    return {
                        "kind": "choice",
                        "content": content,
                        "choices": self._extract_choices(tool_results),
                        "tool_results": tool_results,
                        "data_preview": None,
                        "messages": messages,
                    }

                # 非体检态：正常记录并回灌 LLM
                tool_results.append({
                    "tool": result.get("tool"),
                    "status": result.get("status"),
                    "summary": result.get("summary"),
                    "await_choice": result.get("await_choice", False),
                    "data": result.get("data", {}),
                })
                # 改法1：仅当「本轮执行的 tool」是 clean_data 执行态成功时才置位。
                # 关键：必须用 result（本轮这一个工具的结果），不能用累积的 tool_results，
                # 否则 clean_data=ok 会永远留在 tool_results 里，导致后续每一轮末尾都重新置位 → 重复注入三分析。
                if result.get("tool") == "clean_data" and result.get("status") == "ok" \
                        and not result.get("await_choice"):
                    _need_inject_clean_prompt = True
                    print(f"[agentic_chat] round={_round} clean_data ok (exec) -> set _need_inject_clean_prompt")
                # 工具结果回灌给 LLM
                # 规范：注入「总结义务」+ 从 data 抽出的结论文字，让 LLM 最终必须覆盖该工具结果
                tool_name = tc.function.name
                conclusions = self._extract_conclusions(result.get("data", {}))
                feedback_lines = [
                    f"【工具执行结果】你已通过工具「{tool_name}」完成分析。",
                    "下面是该工具的分析结论，你必须在最终回复中覆盖这个工具的分析结果，不得遗漏。",
                ]
                if conclusions:
                    feedback_lines.append(conclusions)
                feedback_lines.append(
                    "完整数据（供引用细节）："
                    + self._truncate_tool_data(result.get("data", {}))
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "\n".join(feedback_lines),
                })

        # 超过 8 轮仍未结束 → 强制收尾（正常有熔断，通常不会走到这）
        return {
            "kind": "text",
            "content": self._strip_tool_tags(
                "分析步骤较多，已自动终止以避免过长响应。你可以继续追问或分步执行。"
            ),
            "choices": [], "tool_results": tool_results,
            "data_preview": self._build_data_preview(tool_results),
            "messages": messages,
        }

    @staticmethod
    def _truncate_tool_data(data: Any, max_chars: int = 4000) -> str:
        """将工具回灌 data 序列化为 JSON 字符串并做长度截断。

        防止 profile_data 等工具的完整快照（含逐列统计）随每轮回灌不断
        累积进 messages 历史，导致历史膨胀、LLM 更易迷失而反复调工具（大类 D）。
        超限时截断并附提示，结论文字已由 _extract_conclusions 单独抽取回灌，截断不影响总结。
        """
        try:
            text = json.dumps(data, ensure_ascii=False, default=str)
        except Exception:
            text = str(data)
        if len(text) > max_chars:
            return text[:max_chars] + f"…（已截断，原长 {len(text)} 字符，完整结论见上方分析结论）"
        return text

    @staticmethod
    def _extract_conclusions(data: Dict[str, Any]) -> str:
        """从工具 data 抽取分析结论文字，供回灌 LLM 时辅助其写总结。

        业务模型 / 自由写码的结果都在 data.packages[].conclusion；
        通用统计的逐列文字在其 data 结构化字段里（已由回灌 JSON 携带），这里无需特殊处理。
        """
        if not isinstance(data, dict):
            return ""
        packages = data.get("packages") or []
        parts = []
        for pkg in packages:
            if not isinstance(pkg, dict):
                continue
            conclusion = (pkg.get("conclusion") or "").strip()
            if conclusion:
                title = pkg.get("type") or "分析"
                parts.append(f"【{title}】{conclusion}")
        return "\n".join(parts)

    def _force_summary_from_results(self, messages: List[Dict[str, Any]],
                                    ok_results: List[Dict[str, Any]]) -> str:
        """收尾兜底：当 LLM 最终未写文字总结但确有 ok 结果时，补一轮请求强制其写完整总结。

        - 用 messages 深拷贝，绝不污染原 messages（避免写回 session 带人工提示）。
        - tool_choice='none' 且不传 tools，避免再次触发工具调用陷入循环。
        - 失败/仍为空返回 ""，由调用方兜底保持原 content。
        """
        try:
            summary_msgs = copy.deepcopy(messages)
            n = len(ok_results)
            tool_names = "、".join(tr.get("tool", "分析工具") for tr in ok_results)
            summary_msgs.append({
                "role": "user",
                "content": (
                    f"以下是本次已完成的分析结果（共 {n} 个工具：{tool_names}），"
                    "请基于全部结果写一段完整中文总结，必须覆盖每一个工具的分析结论，"
                    "一个都不能遗漏，不要只总结其中一个。直接输出总结文字，无需再调用工具。"
                ),
            })
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=summary_msgs,
                temperature=0.3,
                timeout=120,
                tool_choice="none",
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[agentic_chat] _force_summary_from_results failed: {type(e).__name__}: {e}")
            return ""

    def _snapshot_for_prompt(self, manager, session_id: str) -> str:
        """生成给 LLM 的精简数据快照（来自 data_recon.scan）。

        多表支持：枚举 session 中所有非合并数据集，逐表扫描并标注来源，让 LLM 知道
        当前存在多张表及其列结构；存在关联键时 LLM 可建议合并或用已生成的宽表分析。
        异常不再静默吞掉；scan 失败或 df 为 None 时降级为 data_profile 文本。
        """
        try:
            from src.data_recon import scan
            session = manager.get_session(session_id)
            if session is None:
                return self._snapshot_from_data_profile(manager, session_id)

            non_merged = [
                did for did, ds in session.datasets.items()
                if not getattr(ds, "is_merged", False)
            ]
            if len(non_merged) <= 1:
                # 单表（或仅宽表）：走原逻辑
                df = self._current_df(manager, session_id)
                if df is not None:
                    snap = scan(df)
                    lines = [f"行数={snap['rows']}，列数={snap['column_count']}"]
                    for c in snap["columns_detail"]:
                        line = f"- {c['name']}（{c['kind']}，缺失{c['missing']}）"
                        if c.get("stats"):
                            s = c["stats"]
                            line += f" 范围[{s.get('min')}~{s.get('max')}] 均值{s.get('mean')}"
                        lines.append(line)
                    return "\n".join(lines)
                return self._snapshot_from_data_profile(manager, session_id)

            # 多表：逐表扫描
            blocks: List[str] = []
            for did in non_merged:
                df = manager.get_dataset_df(session_id, did)
                if df is None or df.empty:
                    continue
                fname = session.datasets[did].file_name or did
                snap = scan(df)
                lines = [f"【表：{fname}（dataset_id={did}）】 行数={snap['rows']}，列数={snap['column_count']}"]
                for c in snap["columns_detail"]:
                    line = f"- {c['name']}（{c['kind']}，缺失{c['missing']}）"
                    if c.get("stats"):
                        s = c["stats"]
                        line += f" 范围[{s.get('min')}~{s.get('max')}] 均值{s.get('mean')}"
                    lines.append(line)
                blocks.append("\n".join(lines))
            # 若已存在合并宽表，也提示（分析时优先用宽表）
            merged = [
                ds for ds in session.datasets.values()
                if getattr(ds, "is_merged", False)
            ]
            if merged:
                blocks.append("【已生成合并宽表】可用 merge_tables 或直接在宽表上分析（列名已含来源前缀）。")
            if blocks:
                return "\n\n".join(blocks)
            return self._snapshot_from_data_profile(manager, session_id)
        except Exception as e:
            print(f"[agentic_chat] _snapshot_for_prompt failed: {type(e).__name__}: {e}")
            return self._snapshot_from_data_profile(manager, session_id)

    def _snapshot_from_data_profile(self, manager, session_id: str) -> str:
        """兜底：把 session.data_profile（chat 路由已算好的侦察结果）转成文本快照。

        与 _snapshot_for_prompt 主路径保持字段一致：从 columns_detail 读取
        name/kind/missing，并在首行附 missing_overview 总缺失数，使 LLM 能据
        此判断「是否有缺失值」（这是清洗前置链路的前提）。注意 data_profile['columns']
        是列名字符串列表，真正的列元数据在 columns_detail 里。
        """
        try:
            session = manager.get_session(session_id)
            dp = getattr(session, "data_profile", None)
            if not dp:
                return ""
            rows = dp.get("rows")
            col_count = dp.get("column_count")
            overview = dp.get("missing_overview") or {}
            total_missing = overview.get("total_missing", 0)
            cols_with_missing = overview.get("cols_with_missing", 0)
            cols_detail = dp.get("columns_detail") or []
            lines = [
                f"行数={rows}，列数={col_count}，"
                f"总缺失值={total_missing}（涉及 {cols_with_missing} 列）"
            ]
            for c in cols_detail:
                name = c.get("name", "?")
                kind = c.get("kind", "")
                missing = c.get("missing", 0)
                missing_pct = c.get("missing_pct", 0.0)
                if missing and missing > 0:
                    lines.append(f"- {name}（{kind}，缺失{missing}，占比{missing_pct:.1f}%）")
                else:
                    lines.append(f"- {name}（{kind}，无缺失）")
            return "\n".join(lines)
        except Exception as e:
            print(f"[agentic_chat] _snapshot_from_data_profile failed: {type(e).__name__}: {e}")
            return ""

    @staticmethod
    def _is_complete_summary(content: str, ok_results: List[Dict[str, Any]]) -> bool:        """启发式判断 LLM 的 content 是否已覆盖全部已执行工具的结论。

        判断依据（任一满足即认为不完整，需要续接）：
          1. content 过短（< 80 字）—— 明显只是建议/截断，没写完整总结；
          2. 没有收尾词（总结/结论/综上/分析如下/整体/建议：）—— 说明还没到总结段；
          3. 工具结论里的关键数字/关键词完全没出现在 content 中（缺覆盖）。
        注意：这是兜底续接，配合 _MAX_CONTINUE 上限避免死循环。
        """
        if not content or not content.strip():
            return False
        text = content.strip()
        # 1) 过短直接认为不完整
        if len(text) < 80:
            return False
        # 2) 收尾词缺失 → 视为未到总结段
        closing = ("总结", "结论", "综上", "分析如下", "整体来看", "总体", "综上所述")
        if not any(k in text for k in closing):
            return False
        # 3) 关键结论关键词覆盖检查：从工具结果抽几个代表 token，看 content 是否提及
        missing = 0
        for tr in ok_results:
            toks = _extract_key_tokens(tr.get("data", {}))
            if toks:
                # 该工具的前 3 个 token 中至少命中 1 个，才算被覆盖
                hits = sum(1 for t in toks[:3] if t and t in text)
                if hits == 0:
                    missing += 1
        # 超过半数工具结论未被覆盖 → 认为不完整
        if ok_results and missing > len(ok_results) / 2:
            return False
        return True

    @staticmethod
    def _extract_key_tokens(data: Dict[str, Any]) -> List[str]:
        """从工具结果 data 抽取代表关键词（用于判断总结是否覆盖该工具结论）。"""
        toks: List[str] = []
        try:
            # 业务模型 / 自由写码：packages[].conclusion 里的名词
            for pkg in (data.get("packages") or []):
                concl = pkg.get("conclusion") or ""
                # 取结论里出现的中文数字/百分比/短词片段（粗粒度）
                for seg in str(concl).split():
                    seg = seg.strip("，。、：；()（）")
                    if 2 <= len(seg) <= 8:
                        toks.append(seg)
            # 图表：chart_type
            if data.get("chart_type"):
                toks.append(str(data.get("chart_type")))
            # 通用：前几个非空的 str 值
            for v in (data.get("rows") or [])[:3]:
                if isinstance(v, str) and 1 < len(v) < 10:
                    toks.append(v)
        except Exception:
            pass
        # 去重保序
        seen = set()
        out = []
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out[:6]

    @staticmethod
    def _strip_tool_tags(content: str) -> str:
        """兜底清洗：剥离 LLM 可能写进回复的工具调用/结果标签及其内部内容。

        防止 LLM 不守 prompt 时把 <tool_call>...</tool_call>、<tool_result>...</tool_result>
        这类文本（含原始 JSON/字段清单）直接甩给用户。仅匹配已知工具名的标签，
        避免误伤用户正常对话里出现的字面量。
        """
        if not content:
            return content
        import re
        known = r"(?:profile_data|clean_data|run_template|run_python)"
        pattern = re.compile(
            r"<tool_call>\s*(" + known + r")\b.*?</tool_call>"
            r"|<tool_result>.*?</tool_result>",
            re.DOTALL | re.IGNORECASE,
        )
        cleaned = pattern.sub("", content)
        return cleaned.strip()

    def _classify_response(self, content: str, tool_results: List[Dict[str, Any]]) -> str:
        """判断返回类型：若本轮有清洗建议在等用户选择 → choice，否则 text。

        优先级：clean_data 体检态返回里的 await_choice 是最可靠信号（系统明确在等用户选
        填充方式），直接判定 choice；仅在无 await_choice 时，回退到摘要含"建议/缺失"的字眼判断。
        """
        for tr in tool_results:
            if tr.get("tool") == "clean_data" and tr.get("status") == "ok":
                # 第一优先：系统已明确 await_choice（体检态无 method 调用）
                if tr.get("await_choice"):
                    return "choice"
                # 回退：摘要语境含清洗建议关键词
                if "建议" in (tr.get("summary") or "") or "缺失" in (tr.get("summary") or ""):
                    return "choice"
        return "text"

    def _extract_choices(self, tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从清洗体检态结果提取前端选择按钮（4 种填充方法）。

        防御：只接受 method 为字符串且非空的备选项；若 method 是对象/数组/None，
        直接跳过该备选项（不生成 [object Object] 垃圾 id），避免前端收到后
        发给后端导致 LLM 无法解析。
        """
        for tr in tool_results:
            if tr.get("tool") == "clean_data" and tr.get("status") == "ok":
                data = tr.get("data") or {}
                alts = data.get("available_alternatives") or data.get("recommendation", {}).get("alternatives")
                print(f"[DEBUG _extract_choices] raw alts={alts!r}")
                if alts:
                    result = [
                        {"id": a["method"], "label": a.get("label"),
                         "description": a.get("description", "")}
                        for a in alts
                        if isinstance(a.get("method"), str) and a.get("method")
                    ]
                    print(f"[DEBUG _extract_choices] produced ids={[r['id'] for r in result]} types={[type(r['id']).__name__ for r in result]}")
                    return result
        print(f"[DEBUG _extract_choices] no clean_data ok result -> return []")
        return []

    def _build_data_preview(self, tool_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """清洗执行成功后，取清洗后宽表的预览（前若干行）。"""
        for tr in tool_results:
            if tr.get("tool") == "clean_data" and tr.get("status") == "ok":
                data = tr.get("data") or {}
                if data.get("preview_cols"):
                    return {
                        "rows": data.get("preview_rows"),
                        "columns": data.get("preview_cols"),
                        "head": [],  # 实际 head 由路由取最新 merged df 填充
                    }
        return None

    def generate_report(
        self,
        df: pd.DataFrame,
        saved_charts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """生成完整的结构化数据分析报告（五阶段流水线）

        阶段1-3由 report_analyzer.run_full_analysis() 完成（pandas 精确计算）
        阶段4-5由 LLM 完成（洞察生成 + 报告撰写）

        返回 Dict 包含 sections 列表，每个 section 有 type / title / content / insights
        """
        import json as _json
        import traceback as _tb

        try:
            # ---- 阶段 1-3：Python 统计分析 ----
            analysis_data = run_full_analysis(df, saved_charts)
            fields = analysis_data["phase_1_fields"]
            stats = analysis_data["phase_3_stats"]
            charts = analysis_data["phase_2_charts"]

            # ---- 格式化统计结果为可读文本 ----
            overview_text = _format_overview(stats["overview"])

            basic_stats_text = _format_basic_stats(stats["basic_stats"])

            trend_text = _format_trend(stats["trend_analysis"])

            yoy_text = _format_yoy(stats["yoy_mom"])

            top_text = _format_top(stats["top_analysis"])

            structure_text = _format_structure(stats["structure_analysis"])

            anomaly_text = _format_anomalies(stats["anomaly_analysis"])

            charts_text = "\n".join(
                f"- [{c['type']}] {c['title']}（X: {c.get('x','')}, Y: {str(c.get('y',''))}）→ {c.get('reason','')}"
                for c in charts
            ) if charts else "（无推荐图表）"

            # ---- 阶段 4-5：AI 生成洞察和报告 ----
            user_prompt = REPORT_USER_PROMPT_TEMPLATE.format(
                data_overview=overview_text,
                time_dimension=fields.get("time_dimension") or "无",
                metrics=", ".join(fields.get("metrics", [])) or "无",
                dimensions=", ".join(fields.get("dimensions", [])) or "无",
                basic_stats=basic_stats_text,
                trend_analysis=trend_text,
                yoy_mom=yoy_text,
                top_analysis=top_text,
                structure_analysis=structure_text,
                anomaly_analysis=anomaly_text,
                planned_charts=charts_text,
            )

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.4,
                    max_tokens=8192,
                    timeout=120,
                )

                ai_text = response.choices[0].message.content or ""
                # debug: AI text length logged via logging

                # 尝试解析 JSON
                sections, report_title = _parse_report_json(ai_text)
                # 归一化 type 名（_fill_missing_sections 可能注入了新版名）→ 旧版名，确保 _bind_core_charts 能匹配
                sections = _normalize_section_types(sections)
                # debug: parsed sections logged via logging

                # 自动绑定保底图表到对应 section（AI 漏填 chartIndex 时兜底）
                sections = _bind_core_charts_to_sections(sections, charts)

                return {
                    "success": True,
                    "sections": sections,
                    "report_title": report_title,
                    "raw_analysis": analysis_data,
                }

            except Exception as e:
                # 降级：返回纯统计分析数据（不带 AI 洞察）
                return {
                    "success": True,
                    "sections": _normalize_section_types(_build_fallback_sections(analysis_data)),
                    "raw_analysis": analysis_data,
                    "warning": f"AI 生成洞察失败（{str(e)}），仅返回统计数据",
                }

        except Exception as e:
            # 阶段 1-3 或格式化过程出错，打印完整错误到控制台
            import logging as _logging; _logging.getLogger("agent").error(f"generate_report: {e}")
            _tb.print_exc()

    # ===== V3：基于 AnalysisPackage 的报告生成 =====
    def generate_report_from_packages(
        self,
        packages: List[Dict[str, Any]],
        data_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """基于已保存的 AnalysisPackage 生成 AI 分析报告

        Report AI 的唯一职责是读取 AnalysisPackage 并组织语言生成专业报告。
        不再重新分析数据，不访问原始 DataFrame。
        """
        import json as _json
        import traceback as _tb

        packages = packages or []

        if not packages:
            return {
                "success": True,
                "sections": _normalize_section_types([{
                    "type": "executive_summary",
                    "title": "执行摘要",
                    "content": "当前没有已保存的分析结果。请先在分析页面执行分析并保存，再生成报告。",
                }]),
                "packages_used": 0,
            }

        builder = ReportBuilder()
        report_input = builder.build_input(packages, data_profile)

        if not report_input["available_sections"]:
            return {
                "success": True,
                "sections": _normalize_section_types([{
                    "type": "executive_summary",
                    "title": "执行摘要",
                    "content": "已保存的分析包中没有可报告的数据。",
                }]),
                "packages_used": len(packages),
            }

        user_prompt = REPORT_BI_USER_PROMPT_TEMPLATE.format(
            packages_summary=report_input["packages_summary"],
            prompt_text=report_input["prompt_text"],
        )

        warning: Optional[str] = None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REPORT_BI_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=16000,  # 提升上限：全模块报告易超 8192 被截断（用户实测截断）
                timeout=240,  # 与 client 级一致；SDK 重试已关，最坏 240s 即走 fallback
            )

            ai_text = response.choices[0].message.content or ""
            sections, report_title = _parse_report_json(ai_text)

            # 高危发现防漏守卫：保证 CRITICAL/HIGH 发现的图表+文字不被 LLM 漏掉
            sections = _enforce_high_severity_coverage(sections, report_input["sections_data"])

            # 绑定图表信息到 sections
            sections = _bind_package_charts_to_sections(sections, report_input["sections_data"])

            # 归一化 type 名 → 前端兼容格式
            sections = _normalize_section_types(sections)

            return {
                "success": True,
                "sections": sections,
                "report_title": report_title,
                "packages_used": len(packages),
                # 成功路径也返回结构化 degradation（degraded=False），让前端判断逻辑统一
                "degradation": {
                    "degraded": False,
                    "reason": "ok",
                    "message": "",
                    "canRegenerate": False,
                },
            }

        except Exception as e:
            import logging as _logging; _logging.getLogger("agent").warning(f"report fallback: {e}")
            _tb.print_exc()
            warning = f"AI 报告生成失败（{str(e)}），以下为已有分析数据的直接汇总。"

            # 解析异常类型，构造对用户友好的结构化降级说明（归因 AI 接口，不泄露原始堆栈）
            err_text = str(e).lower()
            if "timeout" in err_text or "timed out" in err_text:
                reason = "llm_timeout"
                message = "检测到 AI 接口响应超时，本报告已自动降级为纯统计摘要。您的统计数据完整准确，仅缺少 AI 智能解读，可稍后点击「重新生成（AI 洞察版）」重试。"
            elif any(k in err_text for k in ("connection", "connect", "econn", "unreachable", "refused", "reset")):
                reason = "llm_unavailable"
                message = "检测到 AI 接口暂时不可达，本报告已自动降级为纯统计摘要。您的统计数据完整准确，仅缺少 AI 智能解读，可稍后点击「重新生成（AI 洞察版）」重试。"
            else:
                reason = "llm_error"
                message = "AI 服务暂不可用，本报告已自动降级为纯统计摘要。您的统计数据完整准确，仅缺少 AI 智能解读，可稍后点击「重新生成（AI 洞察版）」重试。"

            try:
                fallback_sections = _build_fallback_from_packages(packages, report_input)
                fallback_sections = _enforce_high_severity_coverage(fallback_sections, report_input["sections_data"])
                # 归一化 type 名 → 前端兼容格式
                fallback_sections = _normalize_section_types(fallback_sections)
                return {
                    "success": True,
                    "sections": fallback_sections,
                    "report_title": "",  # 降级路径无 LLM 标题，前端用默认文案
                    "packages_used": len(packages),
                    "warning": warning,
                    "degradation": {
                        "degraded": True,
                        "reason": reason,
                        "message": message,
                        "canRegenerate": True,
                    },
                }
            except Exception as fb_e:
                return {
                    "success": False,
                    "sections": [],
                    "packages_used": len(packages),
                    "warning": f"报告生成失败：{str(fb_e)}",
                }




def _chart_is_suspect(chart: Dict[str, Any]) -> bool:
    """校验 run_python 返回的 LLM 简单格式图表数据是否可疑（未聚合/结构错乱）。

    LLM 约定 chart 结构（见 prompts.py）：
      bar/line:  {"chart_type": "bar"/"line", "x": [类目...], "y": [数值...]}
      pie/ranking/table: {"chart_type": ..., "data": [...]}

    典型坏 case：LLM 未按 x 聚合，直接把原始流水塞进 y，导致
    len(y) 远大于 len(x)、数值糊成超长未聚合流水（如 19691958574052734）。

    仅做结构性校验（x/y 长度不一致），不做数值量级判断（避免误伤大额金融数据）。
    """
    if not isinstance(chart, dict):
        return False
    ct = str(chart.get("chart_type", "")).lower()
    x = chart.get("x")
    y = chart.get("y")
    # 仅 bar/line 用 x+y；pie/ranking/table 用 data，不在此校验
    if ct not in ("bar", "line"):
        return False
    if not isinstance(x, (list, tuple)) or not isinstance(y, (list, tuple)):
        return False
    # x/y 长度不一致 → 未按类目聚合，判定可疑
    return len(x) != len(y)


def _execute_code_structured(code: str, df, timeout_sec: int = 18) -> Dict[str, Any]:
    """受限沙箱执行 LLM 生成的 Python 代码，返回结构化结果。

    返回 {"text": str, "chart": Optional[dict]} 或 {"error": str}。
    这是模块级函数（不带 self），供 tools_registry.run_python 跨模块调用。

    安全策略（方案2，同进程受限 globals）：
    - AST 预检：禁危险名/危险方法、import 仅放行白名单库；
    - exec_globals 仅注入 df/pd/np + 白名单内置函数（禁 __import__）；
    - 超时使用 daemon 线程 join，不建子进程、不碰 syscall；
    - 使用 df.copy() 防止污染原始数据。
    """
    # 0) 预处理：剥离 LLM 冗余写出的 import（pd/np 已由沙箱注入，写 import 只会触发误拦）。
    #    危险模块的 import（os/sys/subprocess 等）仍在此拦截，不降低安全性。
    code, strip_err = _strip_imports(code)
    if strip_err is not None:
        logger.warning("run_python 沙箱拦截(import): 原因=%s | 代码=\n%s", strip_err, code)
        return {"error": f"[沙箱拦截] {strip_err}"}

    # 1) AST 预检（exec 前拦截）
    pre = _SAFE_EXEC_AST_CHECK(code)
    if pre is not None:
        logger.warning("run_python 沙箱拦截: 原因=%s | 代码=\n%s", pre, code)
        return {"error": f"[沙箱拦截] {pre}"}

    result_container: Dict[str, Any] = {"text": "", "chart": None, "error": None, "done": False}

    def _run():
        try:
            import numpy as np  # ⚠️ agent.py 顶部未 import numpy，沙箱内必须本地导入
            local_vars = {"df": df.copy(), "pd": pd, "np": np}
            # 受控内置函数白名单（禁用 __import__ 等）
            safe_builtins = {
                "print": print, "len": len, "range": range, "str": str, "int": int,
                "float": float, "list": list, "dict": dict, "set": set, "tuple": tuple,
                "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
                "sorted": sorted, "zip": zip, "enumerate": enumerate, "bool": bool,
                "type": type, "isinstance": isinstance, "format": format, "repr": repr,
                "chr": chr, "ord": ord, "any": any, "all": all, "map": map, "filter": filter,
            }
            exec_globals = {"__builtins__": safe_builtins}

            # 允许 LLM 的 import 白名单库在沙箱内可用
            exec(code, exec_globals, local_vars)

            raw = local_vars.get("result", None)
            if raw is None:
                result_container["text"] = "代码执行成功，但未返回 result 变量。"
            elif isinstance(raw, dict) and "text" in raw:
                result_container["text"] = str(raw.get("text", ""))
                chart = raw.get("chart", None)
                if isinstance(chart, dict) and chart.get("chart_type"):
                    # ★ 数值合理性校验：LLM 未按类目聚合（x/y 长度不一致）时，
                    #   丢弃可疑 chart（置空），避免前端渲染出未聚合的乱码柱状图；
                    #   文字结论 text 仍保留，用户仍能看到分析结论。
                    if _chart_is_suspect(chart):
                        logger.warning(
                            "run_python 返回可疑图表数据（x/y 长度不一致，疑似未聚合），"
                            "已丢弃该图：chart_type=%s x_len=%s y_len=%s",
                            chart.get("chart_type"),
                            len(chart.get("x")) if isinstance(chart.get("x"), (list, tuple)) else "?",
                            len(chart.get("y")) if isinstance(chart.get("y"), (list, tuple)) else "?",
                        )
                    else:
                        result_container["chart"] = chart
            else:
                result_container["text"] = str(raw)
        except Exception as e:
            result_container["error"] = f"代码执行出错：{str(e)}"
        finally:
            result_container["done"] = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if not result_container["done"]:
        return {"error": f"[WARN] 代码执行超时（>{timeout_sec}秒），已跳过。"}
    if result_container["error"]:
        return {"error": result_container["error"]}
    return {"text": result_container["text"], "chart": result_container["chart"]}


def _strip_imports(code: str) -> Tuple[str, Optional[str]]:
    """剥离 LLM 冗余写出的 import 语句，返回 (清理后代码, 错误)。

    沙箱已注入 df/pd/np，LLM 习惯写 `import pandas as pd` / `import numpy as np`，
    原逻辑一律硬拦 → 每次自动三分析都先 fail 一次再重试，浪费一轮。

    改为：非危险模块的 import 直接剥掉（删 AST 节点后重编译），代码照常执行；
    危险模块的 import（os/sys/subprocess/shutil/socket/requests/urllib 等）仍拦截，
    安全性不降。

    返回 (code, None) 表示成功（可能原样返回），(原code, 错误串) 表示拦截。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # 语法错误留给后续 AST 预检/exec 报，这里不动
        return code, None

    DANGEROUS_MODULES = {
        "os", "sys", "subprocess", "shutil", "pathlib", "socket",
        "requests", "urllib", "http", "ctypes", "multiprocessing",
        "threading", "pickle", "shelve", "tempfile", "glob",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in DANGEROUS_MODULES:
                    return code, (
                        f"沙箱禁止导入危险模块：{alias.name}。"
                        "仅允许使用已注入的 df/pd/np 做数据分析。"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in DANGEROUS_MODULES:
                return code, (
                    f"沙箱禁止从危险模块导入：{node.module}。"
                    "仅允许使用已注入的 df/pd/np 做数据分析。"
                )

    # 非危险 import：从 AST 中删除顶层 import 节点（不碰函数/类内部的 import）
    stripped = [
        n for n in tree.body
        if not isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    if len(stripped) == len(tree.body):
        # 无 import，原样返回
        return code, None
    new_tree = ast.Module(body=stripped, type_ignores=[])
    try:
        return _ast_to_code(new_tree), None
    except Exception:
        # 重编译失败则回退原代码（交给后续预检/exec 报错）
        return code, None


def _ast_to_code(tree: ast.AST) -> str:
    """把 AST 模块还原成源码字符串（用 ast.unparse，Python3.9+）。"""
    try:
        return ast.unparse(tree)
    except Exception:
        return ast.dump(tree)


def _SAFE_EXEC_AST_CHECK(code: str) -> Optional[str]:
    """模块级 AST 预检（供 _execute_code_structured 调用，避免依赖类方法）。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"代码语法错误：{str(e)}"

    DANGEROUS_NAMES = {
        "os", "subprocess", "sys", "shutil", "pathlib", "socket", "requests",
        "urllib", "http", "open", "eval", "exec", "compile", "__import__",
        "exit", "quit", "input", "breakpoint", "globals", "locals", "vars",
        "getattr", "setattr", "delattr", "memoryview",
    }
    IMPORT_BLOCK_MSG = (
        "沙箱禁止任何 import 语句。pandas 已注入为 `pd`、numpy 已注入为 `np`、"
        "数据已注入为 `df`，请直接使用这些变量，不要写 import。"
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in DANGEROUS_NAMES:
            return f"禁止使用的名称：{node.id}（沙箱不允许操作系统/动态执行等危险调用）"
        if isinstance(node, ast.Attribute):
            if node.attr in {"system", "popen", "remove", "rmdir", "unlink",
                              "rename", "rmtree", "chmod", "kill", "call"}:
                return f"禁止调用的危险方法：{node.attr}"
        if isinstance(node, ast.Import):
            logger.warning("run_python import被拦截(一律禁止): %s | 代码=\n%s",
                           ", ".join(a.name for a in node.names), code)
            return f"[沙箱拦截] {IMPORT_BLOCK_MSG}"
        if isinstance(node, ast.ImportFrom):
            logger.warning("run_python import被拦截(一律禁止): %s | 代码=\n%s",
                           node.module, code)
            return f"[沙箱拦截] {IMPORT_BLOCK_MSG}"
    return None


# ============================================================
# 报告格式化辅助函数
# ============================================================

def _format_overview(ov: Dict[str, Any]) -> str:
    """格式化数据概览"""
    return (
        f"数据行数：{ov['total_rows']:,} 行\n"
        f"数据列数：{ov['total_cols']} 列\n"
        f"列名：{', '.join(ov['column_names'][:15])}"
        f"{'...' if len(ov['column_names']) > 15 else ''}\n"
        f"缺失值：{ov['missing_total']} 个（{ov['missing_rate']}%）\n"
        f"重复行：{ov['duplicate_rows']} 行\n"
        f"数值列：{ov['numeric_columns']} 个，分类列：{ov['categorical_columns']} 个\n"
        f"内存占用：{ov['memory_mb']} MB"
    )


def _format_basic_stats(bs: Dict[str, Any]) -> str:
    """格式化基础统计"""
    lines = []
    for col, s in bs.items():
        lines.append(
            f"【{col}】 总值={s['total']:,.2f}  均值={s['mean']:,.2f}  "
            f"中位数={s['median']:,.2f}  最大值={s['max']:,.2f}  "
            f"最小值={s['min']:,.2f}  标准差={s['std']:,.2f}  样本数={s['count']}"
        )
    return "\n".join(lines) if lines else "（无数值指标）"


def _format_trend(tr: Dict[str, Any]) -> str:
    """格式化趋势分析"""
    lines = []
    for col, t in tr.items():
        g = t.get("overall_growth_rate")
        g_str = f"{g:+.2f}%" if g is not None else "N/A"
        lines.append(
            f"【{col}】 周期数={t['period_count']}  "
            f"首值={t['first_value']:,.2f} → 末值={t['last_value']:,.2f}  "
            f"整体增长率={g_str}  方向={t['direction']}  "
            f"波动率(CV)={t['volatility_cv']:.2f}%  "
            f"最大单次增长={t['max_single_growth']:+.2f}%  "
            f"最大单次下降={t['max_single_decline']:+.2f}%  "
            f"最长连续涨={t['consecutive_up']}次  最长连续跌={t['consecutive_down']}次"
        )
    return "\n".join(lines) if lines else "（无趋势数据）"


def _format_yoy(ym: Dict[str, Any]) -> str:
    """格式化同环比"""
    if not ym.get("has_yoy") and not ym.get("computed"):
        return "（无同环比数据，可能缺少时间维度或年份数据不足）"
    lines = []
    for d in ym.get("details", []):
        lines.append(
            f"【{d['title']}】 指标列={d['value_column']}  "
            f"当前年={d['current_year']}  对比年={d['previous_year']}  "
            f"数据行数={d['row_count']}  含同比={d['has_yoy']}"
        )
    if ym.get("computed"):
        c = ym["computed"]
        lines.append(f"【{c['metric']}】 总计={c['total']:,.2f}  均值={c['mean']:,.2f} ({c['note']})")
    return "\n".join(lines)


def _format_top(ta: Dict[str, Any]) -> str:
    """格式化 Top/Bottom 分析"""
    lines = []
    for key, t in ta.items():
        top_items = ", ".join(f"{k}:{v:,.2f}" for k, v in t.get("top5", {}).items())
        bottom_items = ", ".join(f"{k}:{v:,.2f}" for k, v in t.get("bottom5", {}).items())
        lines.append(
            f"【{key}】 总分类={t['total_categories']}  "
            f"Top1={t['max_category']}({t['max_value']:,.2f})  "
            f"Bottom1={t['min_category']}({t['min_value']:,.2f})  "
            f"Top3集中度={t['top3_concentration']:.1f}%  "
            f"Top5: {top_items}  "
            f"Bottom5: {bottom_items}"
        )
    return "\n".join(lines) if lines else "（无分类维度）"


def _format_structure(sd: Dict[str, Any]) -> str:
    """格式化结构分析"""
    lines = []
    for key, s in sd.items():
        dist = ", ".join(f"{k}:{int(v['share'])}%" for k, v in list(s.get("distribution", {}).items())[:5])
        lines.append(
            f"【{key}】 分类数={s['category_count']}  "
            f"Top3占比={s['top3_share']:.1f}%  "
            f"分布: {dist}"
        )
    return "\n".join(lines) if lines else "（无分类维度）"


def _format_anomalies(al: List[Dict[str, Any]]) -> str:
    """格式化异常分析"""
    if not al:
        return "（未检测到显著异常）"
    lines = []
    for a in al:
        t = a.get("type", "")
        if t == "离群点":
            details = ", ".join(f"{k}(值={v['value']}, Z={v['z_score']})" for k, v in a.get("details", {}).items())
            lines.append(f"【{t}】指标={a.get('metric','')}  规则={a.get('rule','')}  明细: {details}")
        elif t == "IQR异常":
            lines.append(
                f"【{t}】指标={a.get('metric','')}  规则={a.get('rule','')}  "
                f"异常数={a.get('count',0)}/{a.get('total',0)}({a.get('anomaly_rate',0)}%)"
            )
        elif t == "占比异常":
            lines.append(f"【{t}】维度={a.get('dimension','')}  指标={a.get('metric','')}  警告={a.get('warning','')}")
    return "\n".join(lines) if lines else "（未检测到显著异常）"



def _fill_missing_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """补全缺失的 report sections，确保报告结构完整

    注意：required_types 必须与 REPORT_BI_SYSTEM_PROMPT 中定义的 section type 枚举保持一致，
    否则会把 AI 已经正常输出的 section 误判为"缺失"，注入"数据不足"占位。
    """
    existing_types = {s.get("type", "") for s in sections}
    # 与 prompts.py 中 REPORT_BI_SYSTEM_PROMPT 的 section type 枚举对齐
    required_types = [
        "executive_summary", "data_overview", "trend_analysis", "ranking_analysis",
        "structure_analysis", "concentration_analysis", "distribution_analysis",
        "correlation_analysis", "comparison_analysis", "geo_analysis",
        "retention_analysis", "anomaly_analysis", "proportion_analysis",
        "risk_analysis", "management_suggestions", "conclusion",
    ]

    new_sections = []
    for rt in required_types:
        if rt in existing_types:
            continue
        # 缺失章节：插入明确的"未提供"占位，而不是误导性的"数据不足"
        if rt == "executive_summary":
            new_sections.append({
                "type": "executive_summary", "title": "执行摘要",
                "content": "AI 未生成执行摘要。", "chart_titles": [],
            })
        elif rt == "data_overview":
            new_sections.append({
                "type": "data_overview", "title": "数据概览",
                "content": "AI 未生成数据概览。", "chart_titles": [],
            })
        elif rt == "conclusion":
            new_sections.append({
                "type": "conclusion", "title": "总结",
                "content": "AI 未生成总结。", "chart_titles": [],
            })
        elif rt == "management_suggestions":
            new_sections.append({
                "type": "management_suggestions", "title": "管理建议",
                "content": "AI 未生成管理建议。", "chart_titles": [],
            })
        else:
            new_sections.append({
                "type": rt,
                "title": _section_title_for(rt),
                "content": f"{_section_title_for(rt)}：本章节无相关数据。",
                "chart_titles": [],
            })

    if new_sections:
        sections.extend(new_sections)
    return sections


def _section_title_for(section_type: str) -> str:
    """section type → 中文标题"""
    return {
        "trend_analysis": "趋势分析",
        "ranking_analysis": "排名分析",
        "structure_analysis": "结构分析",
        "concentration_analysis": "集中度分析",
        "distribution_analysis": "分布分析",
        "correlation_analysis": "相关性分析",
        "comparison_analysis": "对比分析",
        "geo_analysis": "地理空间分析",
        "retention_analysis": "留存分析",
        "anomaly_analysis": "异常分析",
        "proportion_analysis": "占比分析",
        "risk_analysis": "风险分析",
    }.get(section_type, section_type)

def _parse_report_json(ai_text: str) -> tuple:
    """从 AI 返回的文本中解析 JSON，返回 (sections, report_title)"""
    import json as _json

    # 提取 JSON 块
    if "```json" in ai_text:
        start = ai_text.find("```json") + 7
        end = ai_text.find("```", start)
        json_str = ai_text[start:end].strip()
    elif "```" in ai_text:
        start = ai_text.find("```") + 3
        end = ai_text.find("```", start)
        json_str = ai_text[start:end].strip()
    else:
        # 尝试找到 JSON 对象
        brace_start = ai_text.find("{")
        brace_end = ai_text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            json_str = ai_text[brace_start:brace_end + 1]
        else:
            return [{"type": "error", "title": "AI 返回解析失败", "content": ai_text[:500]}], ""

    try:

        data = _json.loads(json_str)
        sections = data.get("sections", [])
        # 兜底：只在 sections 为空（AI 完全没生成）时才补全。
        # 旧逻辑 len < 5 触发补全会与正常 AI 输出冲突。
        if not sections:
            sections = _fill_missing_sections(sections)
        return sections, data.get("report_title", "")
    except Exception:
        # JSON 解析失败，返回原始文本
        return [{"type": "error", "title": "AI 返回格式异常", "content": ai_text[:1000]}], ""


def _bind_core_charts_to_sections(
    sections: List[Dict[str, Any]],
    charts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """自动将保底图表绑定到对应类型的报告 section 上

    plan_charts 的阶段 A 给每个保底图打了 section 标签（trend/structure/top）。
    此函数确保这些图的 chartIndex 被绑定到对应的 section，AI 即使漏填也能兜底。
    只在 AI 没有填 chartIndex 时才补，不覆盖 AI 已有的绑定。
    """
    # 建立 section.tag → chart 索引的映射
    section_to_chart_idx: Dict[str, int] = {}
    for i, chart in enumerate(charts):
        tag = chart.get("section")
        if tag and tag not in section_to_chart_idx:
            section_to_chart_idx[tag] = i

    for section in sections:
        stype = section.get("type", "")
        # 只在 section 没有 chartIndex 时才补
        if "chartIndex" not in section and stype in section_to_chart_idx:
            section["chartIndex"] = section_to_chart_idx[stype]

    return sections


def _sections_to_markdown(sections: List[Dict[str, Any]]) -> str:
    """将结构化 sections 列表转换为 Markdown 文本（用于 Streamlit 显示）"""
    lines = []

    # 图标映射
    type_colors = {
        "overview": "[INFO]", "kpi": "[KPI]", "trend": "[TREND]",
        "structure": "[STRUCT]", "top": "[TOP]", "anomaly": "[WARN]",
        "conclusion": "[CONCLUSION]", "suggestions": "[SUGGEST]", "next_steps": "[NEXT]",
        "error": "[ERROR]",
    }

    # 洞察类型标签颜色
    label_colors = {
        "趋势洞察": "[TREND]", "结构洞察": "[STRUCT]", "集中度洞察": "[TOP]",
        "异常洞察": "[WARN]", "风险洞察": "[RISK]",
    }

    for section in sections:
        icon = type_colors.get(section.get("type", ""), "[PIN]")
        title = section.get("title", "")
        lines.append(f"## {icon} {title}")
        lines.append("")

        # content 字段
        if section.get("content"):
            lines.append(section["content"])
            lines.append("")

        # insights 字段
        insights = section.get("insights", [])
        if insights:
            for item in insights:
                if isinstance(item, dict):
                    chart_title = item.get("chart_title", "") or ""
                    analysis = item.get("analysis", "") or ""
                    chart_type = item.get("chart_type") or ""
                    table_type = item.get("table_type") or ""
                    rule_id = item.get("rule_id") or ""
                    insight_label = item.get("insight_label") or ""

                    # 构建规则标签行
                    rule_badge = ""
                    if rule_id or chart_type or table_type or insight_label:
                        badge_parts = []
                        if insight_label:
                            label_icon = label_colors.get(insight_label, "")
                            badge_parts.append(f"{label_icon} {insight_label}")
                        if rule_id:
                            badge_parts.append(rule_id)
                        if chart_type and chart_type != "null":
                            badge_parts.append(f"[KPI] {chart_type}")
                        if table_type and table_type not in ("null", ""):
                            badge_parts.append(f"[INFO] {table_type}")
                        rule_badge = f"*[{' | '.join(badge_parts)}]*  "

                    # 渲染内容
                    if chart_title:
                        lines.append(f"- {rule_badge}**{chart_title}**：{analysis}")
                    else:
                        lines.append(f"- {rule_badge}{analysis}")
                elif isinstance(item, str):
                    lines.append(f"- {item}")
            lines.append("")

        # next_steps section 特殊渲染
        if section.get("type") == "next_steps":
            # ---- 生图计划 ----
            charts_plan = section.get("charts_to_create", [])
            if charts_plan:
                chart_type_cn = {
                    "line": "折线图", "bar": "柱状图", "pie": "饼图",
                    "horizontal_bar": "横向条形图", "stacked_bar": "堆叠柱状图",
                    "scatter": "散点图", "histogram": "直方图", "map_3d": "3D 地图",
                    "table": "数据表格",
                }
                lines.append("### [KPI] 推荐生成的图表")
                lines.append("")
                for c in charts_plan:
                    ctype = c.get("chart_type", "")
                    cname = chart_type_cn.get(ctype, ctype)
                    ctitle = c.get("chart_title", "")
                    guide = c.get("guide", "")
                    value = c.get("value", "")
                    rid = c.get("rule_id", "")
                    xa = c.get("x_axis", "")
                    ya = c.get("y_axis", "")
                    rid_str = f" [{rid}]" if rid else ""
                    lines.append(f"- **{ctitle}**{rid_str} → 创建**{cname}**（X={xa}，Y={ya}）")
                    if value:
                        lines.append(f"  > {value}")
                    if guide:
                        lines.append(f"  > [MOUSE]️ {guide}")
                    lines.append("")
            # ---- 操作清单 ----
            action_items = section.get("action_items", [])
            if action_items:
                lines.append("### [OK] 操作清单")
                lines.append("")
                for a in sorted(action_items, key=lambda x: x.get("priority", 99)):
                    lines.append(f"{a.get('priority', '')}. {a.get('action', '')}")
                lines.append("")

        if section.get("type") == "overview":
            lines.append("---")
        lines.append("")

    return "\n".join(lines).strip()


def _build_fallback_sections(analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """构建降级报告（无 AI 洞察时的统计摘要），包含可执行的生图计划"""
    fields = analysis_data["phase_1_fields"]
    stats = analysis_data["phase_3_stats"]
    charts = analysis_data["phase_2_charts"]
    overview = stats["overview"]

    sections: List[Dict[str, Any]] = []

    # 数据概览
    sections.append({
        "type": "overview",
        "title": "数据概览",
        "content": (
            f"本数据集包含 {overview['total_rows']:,} 行 {overview['total_cols']} 列。"
            f"时间维度：{fields.get('time_dimension') or '无'}。"
            f"核心指标：{', '.join(fields.get('metrics', [])) or '无'}。"
            f"分类维度：{', '.join(fields.get('dimensions', [])) or '无'}。"
            f"数据缺失率 {overview['missing_rate']}%，重复行 {overview['duplicate_rows']} 行。"
        ),
    })

    # KPI 指标（规则10：同环比）
    kpi_insights = []
    for col, s in stats.get("basic_stats", {}).items():
        kpi_insights.append({
            "chart_title": f"{col}核心指标",
            "chart_type": None,
            "table_type": None,
            "rule_id": "规则10",
            "insight_label": "趋势洞察",
            "analysis": f"{col}：总计 {s['total']:,.2f}，均值 {s['mean']:,.2f}，最大值 {s['max']:,.2f}",
        })
    sections.append({"type": "kpi", "title": "核心指标", "insights": kpi_insights})

    # 趋势（规则9）
    trend_insights = []
    for col, t in stats.get("trend_analysis", {}).items():
        g = t.get("overall_growth_rate")
        icon = "[UP]" if (g or 0) > 0 else "[DOWN]" if (g or 0) < 0 else "➖"
        trend_insights.append({
            "chart_title": f"{col}趋势分析",
            "chart_type": "line",
            "table_type": "sort",
            "rule_id": "规则9",
            "insight_label": "趋势洞察",
            "analysis": f"整体增长 {icon} {g:+.2f}%，波动率 {t['volatility_cv']:.2f}%，最长连续涨 {t['consecutive_up']} 次。",
        })
    sections.append({"type": "growth_analysis", "title": "趋势分析", "insights": trend_insights})

    # 结构（规则12）
    struct_insights = []
    for key, s in stats.get("structure_analysis", {}).items():
        struct_insights.append({
            "chart_title": key,
            "chart_type": "pie",
            "table_type": "summary",
            "rule_id": "规则12",
            "insight_label": "结构洞察",
            "analysis": f"共 {s['category_count']} 个分类，Top3 占比 {s['top3_share']:.1f}%。",
        })
    sections.append({"type": "structure_analysis", "title": "结构分析", "insights": struct_insights})

    # Top（规则11）
    top_insights = []
    for key, t in stats.get("top_analysis", {}).items():
        top_insights.append({
            "chart_title": f"{key}排名分析",
            "chart_type": "bar",
            "table_type": "sort",
            "rule_id": "规则11",
            "insight_label": "集中度洞察",
            "analysis": f"{key}：Top1={t['max_category']}({t['max_value']:,.2f})，Top3 集中度={t.get('top3_concentration',0):.1f}%",
        })
    sections.append({"type": "ranking_analysis", "title": "TOP / 集中度分析", "insights": top_insights})

    # 异常
    anomaly_insights = []
    for a in stats.get("anomaly_analysis", []):
        if a.get("type") == "占比异常":
            anomaly_insights.append({
                "chart_title": None, "chart_type": None, "table_type": None,
                "rule_id": None, "insight_label": "风险洞察", "analysis": a.get("warning", ""),
            })
        elif a.get("type") == "IQR异常":
            anomaly_insights.append({
                "chart_title": None, "chart_type": None, "table_type": None,
                "rule_id": None, "insight_label": "异常洞察",
                "analysis": f"{a.get('metric','')}：发现 {a.get('count',0)} 个 IQR 异常值（{a.get('anomaly_rate',0):.1f}%）",
            })
    if not anomaly_insights:
        anomaly_insights = [{
            "chart_title": None, "chart_type": None, "table_type": None,
            "rule_id": None, "insight_label": "异常洞察", "analysis": "未检测到显著异常",
        }]
    sections.append({"type": "anomaly_analysis", "title": "异常分析", "insights": anomaly_insights})

    # 结论
    sections.append({
        "type": "conclusion",
        "title": "核心结论",
        "insights": [
            {"chart_title": None, "chart_type": None, "table_type": None,
             "rule_id": None, "insight_label": None,
             "analysis": f"数据规模 {overview['total_rows']:,} 行，{overview['total_cols']} 个字段"},
            {"chart_title": None, "chart_type": None, "table_type": None,
             "rule_id": None, "insight_label": None,
             "analysis": f"缺失率 {overview['missing_rate']}%（{'偏高，建议关注' if overview['missing_rate'] > 5 else '正常范围'}）"},
            {"chart_title": None, "chart_type": None, "table_type": None,
             "rule_id": None, "insight_label": None,
             "analysis": "以上为自动统计分析结果，开启 AI API Key 可生成更丰富的洞察"},
        ],
    })


    # 建议（更具体的业务建议）
    dims = fields.get("dimensions", [])
    mets = fields.get("metrics", [])
    time_col = fields.get("time_dimension")

    suggestion_items = []
    if dims and mets:
        suggestion_items.append({
            "chart_title": None, "chart_type": None, "table_type": None,
            "rule_id": None, "insight_label": None,
            "analysis": f"重点监控 {dims[0]} 维度的 {mets[0] if mets else '指标'} 变化趋势，按周对比异常波动"
        })
    if time_col and mets:
        suggestion_items.append({
            "chart_title": None, "chart_type": None, "table_type": None,
            "rule_id": None, "insight_label": None,
            "analysis": f"建立 {mets[0]} 的月度环比监控，若环比下降超过 15% 触发预警"
        })
    if overview["missing_rate"] > 5:
        suggestion_items.append({
            "chart_title": None, "chart_type": None, "table_type": None,
            "rule_id": None, "insight_label": None,
            "analysis": f"数据缺失率 {overview['missing_rate']}%，建议排查缺失原因并补充数据"
        })
    # 兜底
    if not suggestion_items:
        suggestion_items = [{
            "chart_title": None, "chart_type": None, "table_type": None,
            "rule_id": None, "insight_label": None,
            "analysis": "定期监控指标变化趋势，及时调整业务策略"
        }]
    sections.append({
        "type": "suggestions",
        "title": "业务建议",
        "insights": suggestion_items,
    })

    # ---- 后续操作：操作清单 ----
    action_items = []
    if dims and mets:
        action_items.append({
            "priority": 1,
            "action": f"优先进入「仪表盘」页面，创建折线图监控 {mets[0]} 随时间的变化趋势"
        })
    if len(dims) >= 2 and mets:
        action_items.append({
            "priority": 2,
            "action": f"创建 {dims[0]}×{dims[1]} 交叉分析图，找出双重维度下的增长驱动因素"
        })
    if len(mets) >= 2:
        action_items.append({
            "priority": 3,
            "action": f"创建 {mets[0]} 与 {mets[1]} 的散点图，探索指标间相关性"
        })
    action_items.append({
        "priority": 99,
        "action": "完成图表创建后，点击「生成报告」按钮生成完整分析报告并导出 PDF"
    })
    if not action_items:
        action_items = [{"priority": 1, "action": "去仪表盘页面创建合适的图表，开始数据可视化分析"}]

    sections.append({
        "type": "next_steps",
        "title": "下一步操作建议",
        "action_items": action_items,
    })

    return sections


# ============================================================
# 数据洞察（用户指定格式）辅助函数
# ============================================================

def _build_insights_data_summary(
    df: pd.DataFrame,
    fields: Dict[str, Any],
    stats: Dict[str, Any],
    charts: List[Dict[str, Any]],
) -> str:
    """构建给 LLM 的数据摘要（用于生成用户指定格式的洞察）"""
    overview = stats["overview"]
    lines = []

    # 一、基本信息
    lines.append("【数据基本信息】")
    lines.append(f"- 行数：{overview['total_rows']:,}")
    lines.append(f"- 列数：{overview['total_cols']}")
    lines.append(f"- 完整列名列表：{overview['column_names']}")
    lines.append(f"- 数据类型：{dict(zip(overview['column_names'], [str(d) for d in df.dtypes]))}")
    lines.append("")

    # 二、字段分类（最重要！LLM 必须使用这些真实列名）
    time_col = fields.get("time_dimension")
    metrics = fields.get("metrics", [])
    dimensions = fields.get("dimensions", [])

    lines.append("【字段分类——分析建议中必须使用这些真实列名！】")
    lines.append(f"- [CAL] 时间列：{time_col if time_col else '（无）'}")
    lines.append(f"- [KPI] 数值指标列：{', '.join(metrics) if metrics else '（无）'}")
    lines.append(f"- [TAG]️ 分类维度列：{', '.join(dimensions) if dimensions else '（无）'}")
    # 识别地区列
    region_cols = [c for c in dimensions if any(
        kw in str(c).lower() for kw in ['省', '市', '区', '县', '地区', '区域', '城市', '省份',
                                          'province', 'city', 'region', 'district', 'area']
    )]
    if region_cols:
        lines.append(f"- [MAP]️ 地区/地图列：{', '.join(region_cols)}（必须推荐 3D 地图！）")
    lines.append("")

    # 三、数据质量
    missing_info = df.isnull().sum().to_dict()
    missing_cols = {k: v for k, v in missing_info.items() if v > 0}
    lines.append("【数据质量】")
    lines.append(f"- 总缺失值：{overview['missing_total']} 个（{overview['missing_rate']}%）")
    if missing_cols:
        lines.append(f"- 有缺失的列：{missing_cols}")
    lines.append(f"- 重复行：{overview['duplicate_rows']} 行")
    lines.append("")

    # 四、数值列统计
    if metrics:
        lines.append("【数值指标统计（用于关键发现中引用具体数字）】")
        for col_name in metrics[:5]:
            if col_name in df.columns:
                col_data = df[col_name].dropna()
                if len(col_data) > 0 and pd.api.types.is_numeric_dtype(col_data):
                    lines.append(
                        f"- {col_name}：均值={col_data.mean():,.2f} "
                        f"中位数={col_data.median():,.2f} "
                        f"总和={col_data.sum():,.2f} "
                        f"最大值={col_data.max():,.2f} "
                        f"最小值={col_data.min():,.2f} "
                        f"标准差={col_data.std():,.2f}"
                    )
        lines.append("")

    # 五、分类列信息
    if dimensions:
        lines.append("【分类维度信息（用于分析建议中的对比/排名/占比分析）】")
        for dim in dimensions[:5]:
            if dim in df.columns:
                vc = df[dim].value_counts()
                top3 = ", ".join(f"{k}({v})" for k, v in vc.head(3).items())
                lines.append(f"- {dim}：共 {len(vc)} 个分类，Top3：{top3}")
        lines.append("")

    # 六、已规划图表（作为参考，LLM 可据此生成分析建议）
    if charts:
        lines.append("【系统已推荐的图表（作为分析建议的参考）】")
        for c in charts[:10]:
            x = c.get('x', '')
            y = c.get('y', '')
            t = c.get('type', '')
            title = c.get('title', '')
            reason = c.get('reason', '')
            lines.append(f"- [{t}] {title}（X={x}, Y={y}）→ {reason}")
        lines.append("")

    return "\n".join(lines)


def _clean_insights_text(ai_text: str) -> str:
    """清洗 AI 返回的洞察文本（去除常见前缀后缀）"""
    text = ai_text.strip()
    # 去掉 AI 可能添加的前导语
    prefixes_to_strip = [
        "好的，以下是对数据的分析：",
        "好的，以下是数据分析报告：",
        "以下是对给定数据的分析：",
        "根据提供的数据，分析如下：",
        "好的，我来分析：",
        "以下是对数据的自动分析：",
    ]
    for prefix in prefixes_to_strip:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # 去掉可能的 markdown code block 包裹
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline > 0:
            text = text[first_newline:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    # 确保以 ## 开头
    if not text.startswith("##") and not text.startswith("##"):
        lines = text.split("\n")
        # 找到第一个 ## 行
        for i, line in enumerate(lines):
            if line.strip().startswith("##"):
                text = "\n".join(lines[i:])
                break
    return text


def _build_fallback_insights(
    df: pd.DataFrame,
    fields: Dict[str, Any],
    stats: Dict[str, Any],
    charts: List[Dict[str, Any]],
    error_msg: str,
) -> str:
    """构建降级洞察（无 LLM 时用 Python 直接生成用户指定格式）"""
    overview = stats["overview"]
    time_col = fields.get("time_dimension")
    metrics = fields.get("metrics", [])
    dimensions = fields.get("dimensions", [])
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in numeric_cols and c != time_col]

    # 识别地区列
    region_cols = [c for c in dimensions if any(
        kw in str(c).lower() for kw in ['省', '市', '区', '县', '地区', '区域', '城市', '省份',
                                          'province', 'city', 'region', 'district', 'area']
    )]

    lines = []

    # ---- 数据概览 ----
    lines.append("## 数据概览")
    lines.append(
        f"本数据集包含 {overview['total_rows']:,} 行、{overview['total_cols']} 列，"
        f"涵盖 {len(metrics)} 个数值指标和 {len(dimensions)} 个分类维度。"
    )
    if time_col:
        lines.append(f"数据有时间维度「{time_col}」，支持趋势分析和同环比计算。")
    if region_cols:
        lines.append(f"数据包含地区维度「{region_cols[0]}」，支持地理分布分析。")
    lines.append("")

    # ---- 关键发现 ----
    lines.append("## 关键发现")
    finding_idx = 1
    # 时间趋势
    if time_col and metrics:
        col = metrics[0]
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            lines.append(f"{finding_idx}. 「{col}」数据整体均值为 {df[col].mean():,.2f}，最高值 {df[col].max():,.2f}，最低值 {df[col].min():,.2f}，标准差 {df[col].std():,.2f}")
            finding_idx += 1
    # Top 集中度
    for dim in dimensions[:2]:
        if dim in df.columns:
            vc = df[dim].value_counts()
            top1 = vc.index[0]
            top3_share = vc.head(3).sum() / vc.sum() * 100 if vc.sum() > 0 else 0
            lines.append(f"{finding_idx}. 「{dim}」维度共 {len(vc)} 个分类，Top1 为「{top1}」，Top3 占比 {top3_share:.1f}%")
            finding_idx += 1
    # 每指标统计
    for col in metrics[1:3]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            lines.append(f"{finding_idx}. 「{col}」总和 {df[col].sum():,.2f}，中位数 {df[col].median():,.2f}，数据离散度为 {df[col].std()/df[col].mean()*100 if df[col].mean() != 0 else 0:.1f}%")
            finding_idx += 1
    # 异常
    anomaly_list = stats.get("anomaly_analysis", [])
    if anomaly_list:
        lines.append(f"{finding_idx}. 检测到 {len(anomaly_list)} 类数据异常，包括：{', '.join(a.get('type', '') for a in anomaly_list)}")
        finding_idx += 1
    if finding_idx <= 1:
        lines.append("1. 数据规模适中，字段结构完整，可进行多维度交叉分析")
    lines.append("")

    # ---- 数据质量 ----
    lines.append("## 数据质量")
    missing_cols = {k: v for k, v in df.isnull().sum().to_dict().items() if v > 0}
    if missing_cols:
        lines.append(f"存在缺失值：{missing_cols}，整体缺失率 {overview['missing_rate']}%。")
    else:
        lines.append("数据完整性良好，无缺失值。")
    if overview['duplicate_rows'] > 0:
        lines.append(f"发现 {overview['duplicate_rows']} 行重复数据，建议去重后分析。")
    # 异常值
    anomaly_list = stats.get("anomaly_analysis", [])
    if anomaly_list:
        for a in anomaly_list[:3]:
            t = a.get("type", "")
            if t == "IQR异常":
                lines.append(f"「{a.get('metric', '')}」列存在 {a.get('count', 0)} 个 IQR 异常值。")
            elif t == "占比异常":
                lines.append(f"「{a.get('dimension', '')}」维度：{a.get('warning', '')}")
    else:
        lines.append("未检测到显著异常值。")
    lines.append("")

    # ---- 分析建议 ----
    lines.append("## 分析建议")
    lines.append("以下建议包含计算列和推荐图表，每条标注X轴列和Y轴列，点击「应用洞察」可自动执行：")
    lines.append("")
    s_idx = 1

    # 策略1：有时间列 + 金额/数值列 → 优先同环比
    if time_col and metrics:
        for metric in metrics[:2]:
            lines.append(f"{s_idx}. 计算「{metric}」的同比（与去年同月对比）→ 折线图（X:{time_col}, Y:{metric}同比）")
            lines.append(f"    + 排序表格（排序:{metric}同比变化%, 降序）")
            s_idx += 1
            lines.append(f"{s_idx}. 计算「{metric}」的环比（与上月对比）→ 折线图（X:{time_col}, Y:{metric}环比）")
            lines.append(f"    + 排序表格（排序:{metric}环比变化%, 降序）")
            s_idx += 1
            lines.append(f"{s_idx}. 计算「{metric}」的累计值 → 面积图（X:{time_col}, Y:{metric}累计）")
            s_idx += 1
            lines.append(f"{s_idx}. 计算「{metric}」的移动平均（3月平滑）→ 折线图（X:{time_col}, Y:{metric}移动平均）")
            s_idx += 1
            break  # 只对第一个指标做同环比

    # 策略2：有地区列 → 3D 地图
    if region_cols and metrics:
        lines.append(f"{s_idx}. 绘制「{region_cols[0]}」的「{metrics[0]}」地图与省份地区分布 → 3D地图（X:{region_cols[0]}, Y:{metrics[0]}）")
        lines.append(f"    + 汇总表格（行:{region_cols[0]}, 列:{metrics[0]}）")
        s_idx += 1

    # 策略3：有分类维度 + 数值指标 → 对比排名 / 占比
    if dimensions and metrics:
        lines.append(f"{s_idx}. 计算各「{dimensions[0]}」的「{metrics[0]}」均值，对比排名 → 柱状图（X:{dimensions[0]}, Y:{metrics[0]}）")
        lines.append(f"    + 排序表格（排序:{metrics[0]}, 降序）")
        s_idx += 1

    # 策略4：≥2 个数值指标 → 散点图
    if len(metrics) >= 2:
        lines.append(f"{s_idx}. 探索「{metrics[0]}」与「{metrics[1]}」的相关与关联关系 → 散点图（X:{metrics[0]}, Y:{metrics[1]}）")
        lines.append(f"    + 相关系数表格（行:{metrics[0]}, 列:{metrics[1]}）")
        s_idx += 1

    # 策略5：数值列分布
    if metrics:
        lines.append(f"{s_idx}. 分析「{metrics[0]}」的分布与频次 → 直方图（X:{metrics[0]}, Y:）")
        s_idx += 1

    # 策略6：≥2 个分类维度 → 交叉分析
    if len(dimensions) >= 2 and metrics:
        lines.append(f"{s_idx}. 计算「{metrics[0]}」按「{dimensions[0]}」×「{dimensions[1]}」的交叉汇总 → 堆叠柱状图（X:{dimensions[0]}, Y:{metrics[0]}, 分组:{dimensions[1]}）")
        lines.append(f"    + 交叉表格（行:{dimensions[0]}, 列:{dimensions[1]}, 值:{metrics[0]}）")
        s_idx += 1

    # 策略7：无时间列时 → 排名
    if not time_col and dimensions and metrics:
        lines.append(f"{s_idx}. 计算「{metrics[0]}」按「{dimensions[0]}」的排名 → 柱状图（X:{dimensions[0]}, Y:{metrics[0]}）")
        lines.append(f"    + 排序表格（排序:{metrics[0]}, 降序）")
        s_idx += 1

    if s_idx == 1:
        # 兜底
        if numeric_cols and cat_cols:
            lines.append(f"1. 计算各「{cat_cols[0]}」的「{numeric_cols[0]}」对比排名 → 柱状图（X:{cat_cols[0]}, Y:{numeric_cols[0]}）")
            lines.append(f"    + 排序表格（排序:{numeric_cols[0]}, 降序）")

    return "\n".join(lines) + f"\n\n\n---\n\n> [WARN] AI 洞察生成失败（{error_msg}），以上为自动统计分析结果。"



# ============================================================
# V3：基于 AnalysisPackage 的报告辅助函数
# ============================================================

# 归一化：将 AI prompt 中使用的 section type（新名）映射到前端 DashboardPage 硬编码的旧名
# 前端 DashboardPage.tsx:417-433 只识别：overview/kpi/trend/top/structure/anomaly/conclusion/suggestions/next_steps
_SECTION_TYPE_NORMALIZE: Dict[str, str] = {
    "data_overview": "overview",
    "trend_analysis": "trend",
    "growth_analysis": "trend",  # 五阶段流水线 _build_fallback_sections 使用的旧名
    "ranking_analysis": "top",
    "structure_analysis": "structure",
    "anomaly_analysis": "anomaly",
    "conclusion": "conclusion",
    "management_suggestions": "suggestions",
    "action_items": "next_steps",
}

def _normalize_section_types(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 AI 返回/fallback 生成的 sections 的 type 转为前端兼容的旧名

    同时为 next_steps 类型的 section 容错：AI 可能输出 actions/steps/recommendations
    等变体字段名，统一规范为 action_items（前端 buildReportHTML 识别的字段名）。
    """
    for s in sections:
        old_type = _SECTION_TYPE_NORMALIZE.get(s.get("type", ""), s.get("type", ""))
        s["type"] = old_type
        # next_steps 容错：合并各种可能的字段名到 action_items
        if s.get("type") == "next_steps" and not s.get("action_items"):
            merged: List[Dict[str, Any]] = []
            for alt_key in ("actions", "steps", "recommendations", "action_list", "items"):
                alt = s.get(alt_key)
                if isinstance(alt, list):
                    for item in alt:
                        if isinstance(item, str):
                            merged.append({"priority": 99, "action": item})
                        elif isinstance(item, dict):
                            merged.append({
                                "priority": item.get("priority", item.get("顺序", 99)),
                                "action": item.get("action", item.get("action_text", item.get("text", str(item)))),
                            })
            if merged:
                s["action_items"] = merged
    return sections

# ===== 高危发现防漏守卫（severity → 挑图 的硬性保证）=====
# 背景：LLM 可能漏写 CRITICAL/HIGH 发现的 chart_title（图缺失）甚至整条发现（文字缺失）。
# 此处确定性补入：直接用发现自身写好的 business_meaning/impact/recommendation 作文字
# （不二次调 LLM、不编造），并将其 evidence.chart_slots → 图表 title 补入对应章节。
# 满足项目铁律「文字优先 / 图表仅作证据」：补入的高危必带文字，不只一张裸图。

# V3 章节名 → LLM 旧章节类型（用于把补入洞察塞到语义匹配的已有章节；无匹配则新建）
_V3_TO_LLM_SECTION = {
    "risk_analysis": "anomaly",
    "retention_analysis": "trend",
    "concentration_analysis": "structure",
    "structure_analysis": "structure",
    "correlation_analysis": "anomaly",
    "funnel_analysis": "structure",
    "geo_analysis": "structure",
}

def _norm_title(s: Any) -> str:
    """图表标题归一化：NFKC 全角转半角 + 去所有空白 + 转小写，用于模糊匹配。

    LLM 常把源标题简写（如 '转化漏斗' vs 源 '转化漏斗（AARRR）'），
    精确串匹配会同时破坏三处：守卫重复补图、兄弟章节匹配、前端图绑定。
    归一化后三者一致，是修复 BUG1 的根基。
    """
    if not s:
        return ""
    return unicodedata.normalize("NFKC", str(s)).replace(" ", "").replace("\u3000", "").lower()


def _title_to_slot(title: Any, slot_to_title: Dict[str, str]):
    """把（可能简写的）图表标题反查到源 slot。返回 slot 或 None。

    匹配优先级：① 归一化精确相等 ② 包含关系（简写⊂源，最常见）
    ③ difflib 相似度 ≥0.85（微小改写如 '八' vs '8'）。
    """
    if not title:
        return None
    nt = _norm_title(title)
    if not nt:
        return None
    best, best_score = None, 0.0
    for slot, src in slot_to_title.items():
        ns = _norm_title(src)
        if not ns:
            continue
        if nt == ns:
            return slot
        if nt in ns or ns in nt:
            return slot
        ratio = difflib.SequenceMatcher(None, nt, ns).ratio()
        if ratio >= 0.85 and ratio > best_score:
            best, best_score = slot, ratio
    return best


def _dedupe_section_insights_by_slot(
    sections: List[Dict[str, Any]], slot_to_title: Dict[str, str]
) -> None:
    """每个 section 内按图表 slot 去重 insights（LLM 简写已对齐，同 slot 必同标题）。

    兜底：即便前面的判断逻辑漏判，补图后也保证同一张图在同一章节只出现一次。
    按 section 维度去重，不影响设计允许的跨章节引用（如 churn_seg 引 kmeans 的图）。
    """
    for s in sections:
        ins_list = s.get("insights")
        if not isinstance(ins_list, list) or not ins_list:
            continue
        seen_slots = set()
        kept = []
        for ins in ins_list:
            if not isinstance(ins, dict):
                kept.append(ins)
                continue
            ct = ins.get("chart_title")
            slot = _title_to_slot(ct, slot_to_title) if ct else None
            if slot is not None:
                if slot in seen_slots:
                    continue  # 同图重复，丢弃多余的一条
                seen_slots.add(slot)
            kept.append(ins)
        s["insights"] = kept


_HIGH_SEVERITY = {"critical", "high"}


def _enforce_high_severity_coverage(
    sections: List[Dict[str, Any]],
    sections_data: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """保证所有 CRITICAL/HIGH 业务发现的『图表 + 文字』都出现在报告中。

    返回（可能已就地修改）sections。
    """
    # 1) 全局 slot -> title / slot -> type 桥（必须在收集 LLM 引用之前构建，
    #    以便把 LLM 简写标题对齐回源精确标题）。
    #    与 _bind_package_charts_to_sections 一致：图表绑定是跨包全局 chart_map，
    #    故守卫也必须全局解析——否则跨包图引用（如 churn_seg 的 finding 引 kmeans 的
    #    cluster_radar）会被误判为不可覆盖的悬空槽，导致高危图防漏失效。
    slot_to_title = {}
    slot_to_type = {}
    for pkgs in sections_data.values():
        for pkg in pkgs:
            for c in pkg.get("chart_data", []):
                if c.get("slot") and c.get("title"):
                    slot_to_title[c["slot"]] = c["title"]
                if c.get("slot") and c.get("type"):
                    slot_to_type[c["slot"]] = c["type"]

    # 2) LLM 已引用的 chart_title / chart_slot 集合；
    #    并就地把 LLM 简写 chart_title 对齐回源精确标题——这一对齐同时修复三处：
    #    (a) 守卫补图判断不再因标题不一致而重复补；(b) _find_sibling_section 能认出
    #    兄弟章节而不新建重复章节；(c) _bind_package_charts_to_sections 能按精确标题
    #    绑定图表，前端图可正常加载。
    referenced_titles = set()
    referenced_slots = set()
    for s in sections:
        ct_list = s.get("chart_titles") or []
        if not isinstance(ct_list, list):
            continue
        for i, ct in enumerate(ct_list):
            if not ct:
                continue
            matched_slot = _title_to_slot(ct, slot_to_title)
            if matched_slot is not None:
                ct_list[i] = slot_to_title[matched_slot]  # 就地对齐回源精确标题
                referenced_slots.add(matched_slot)
            referenced_titles.add(ct_list[i])

    # 3) 逐章节检查高危发现
    for section_name, pkgs in sections_data.items():
        for pkg in pkgs:
            for f in pkg.get("findings", []):
                if not isinstance(f, dict) or str(f.get("severity", "")).lower() not in _HIGH_SEVERITY:
                    continue

                evidence = f.get("evidence") or {}
                slots = evidence.get("chart_slots") or []

                if slots:
                    # 图表型发现：逐张图（按 slot）检查覆盖，缺哪张补哪张。
                    # 用 slot 维度判断而非标题精确匹配——LLM 简写标题已对齐回源，
                    # 即便个别未对齐也能通过 slot 识别是否已引用，杜绝重复补图。
                    # 不能用 any() 短路——否则 LLM 引用了同发现的另一张图，
                    # 会把本张高危图漏掉（实测：churn_rule 一张饼图被引用，
                    # 另一张气泡矩阵图被静默跳过）。
                    for slot in slots:
                        if slot not in slot_to_title:
                            continue
                        t = slot_to_title[slot]
                        if slot in referenced_slots or t in referenced_titles:
                            continue
                        _inject_finding_chart_title(sections, section_name, t, slot_to_title)
                        referenced_slots.add(slot)
                        referenced_titles.add(t)
                else:
                    # 纯文字发现：粗略检查标题是否已在某 section content 文字中出现
                    ftitle = f.get("title", "")
                    if ftitle and _finding_text_covered(sections, ftitle):
                        continue
                    _inject_finding_chart_title(sections, section_name, None, slot_to_title)

    # 4) 兜底：每个 section 内按图表标题去重，确保同一张图绝不在同一章节出现两次
    #    （即便前述判断逻辑存在边界漏判，此处也保证最终报告无重复图）。
    for s in sections:
        ct_list = s.get("chart_titles")
        if isinstance(ct_list, list) and ct_list:
            seen = set()
            kept = []
            for ct in ct_list:
                if ct and ct not in seen:
                    seen.add(ct)
                    kept.append(ct)
            s["chart_titles"] = kept
    return sections


def _finding_text_covered(sections, ftitle):
    for s in sections:
        blob = " ".join(str(s.get(k, "")) for k in ("content", "title"))
        if ftitle and ftitle in blob:
            return True
    return False


def _inject_finding_chart_title(sections, section_name, target_title, slot_to_title):
    """把单个高危发现的图表标题补入报告的 chart_titles（图随其所属章节就近展示）。

    - target_title 为图表标题：补入找到的目标章节的 chart_titles 数组（去重）。
    - target_title 为 None（纯文字发现）：仅确保对应章节存在（依附兄弟章节或新建），
      不补图，符合项目铁律「文字优先」。
    """
    target_section = _find_sibling_section(sections, slot_to_title)
    if target_section is None:
        target_section = _create_section_for(sections, section_name)
    if target_title is None:
        return
    ct_list = target_section.setdefault("chart_titles", [])
    if target_title not in ct_list:
        ct_list.append(target_title)


def _find_sibling_section(sections, slot_to_title):
    """找已存在且包含本章节兄弟图表的 LLM 章节（保持图表在合适位置）。

    改用归一化匹配：LLM 简写标题（如 '转化漏斗'）也能认出对应的兄弟章节，
    避免因精确匹配失败而新建一个重复章节（BUG1 的连锁表现之一）。
    新结构（V4）：兄弟图声明在 section.chart_titles（字符串数组）中。
    """
    sibling_titles_norm = {_norm_title(t) for t in slot_to_title.values() if t}
    if not sibling_titles_norm:
        return None
    for s in sections:
        for ct in (s.get("chart_titles") or []):
            if ct and _norm_title(ct) in sibling_titles_norm:
                return s
    return None


def _create_section_for(sections, section_name):
    """无兄弟章节时新建一个（用 LLM 能渲染的旧类型 + 中文标题）。"""
    llm_type = _V3_TO_LLM_SECTION.get(section_name, "anomaly")
    title = section_name
    try:
        from src.report_builder import SECTION_DISPLAY_NAME
        title = SECTION_DISPLAY_NAME.get(section_name, section_name)
    except Exception:
        pass
    new_section = {"type": llm_type, "title": title, "insights": []}
    sections.append(new_section)
    return new_section


def _category_to_insight_label(category):
    mapping = {
        "RISK": "风险洞察",
        "ANOMALY": "异常洞察",
        "CONCENTRATION": "集中度洞察",
        "CORRELATION": "相关性洞察",
        "STRUCTURE": "结构洞察",
        "COMPOSITION": "结构洞察",
        "COMPARISON": "结构洞察",
        "GROWTH": "趋势洞察",
    }
    return mapping.get(str(category).upper(), "结构洞察")


def _bind_package_charts_to_sections(
    sections: List[Dict[str, Any]],
    sections_data: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """将 AnalysisPackage 中的图表信息绑定到 AI 生成的 sections 中。

    新结构（V4）：每个 section 用 `chart_titles`（字符串数组）声明其正文中引用的图表，
    此处把它们对应的完整图表（含 option/raw_data）提取成 `section_charts` 挂回 section，
    供前端就近插图。不再依赖旧的 insights[].chart_title 结构。
    """
    # 全局 title -> 完整图表对象 映射（图表可能来自跨包，故全局解析）
    chart_map: Dict[str, Dict[str, Any]] = {}
    for pkgs in sections_data.values():
        for pkg in pkgs:
            for c in pkg.get("chart_data", []):
                title = c.get("title", "")
                if title and title not in chart_map:
                    chart_map[title] = c

    for section in sections:
        ct_list = section.get("chart_titles")
        if not isinstance(ct_list, list) or not ct_list:
            section["section_charts"] = []
            continue
        bound = []
        seen_slots = set()
        for ct in ct_list:
            if not ct:
                continue
            # 优先精确匹配，其次归一化包含匹配（LLM 可能简写标题）
            chart = chart_map.get(ct)
            if chart is None:
                nt = _norm_title(ct)
                for src_title, src_chart in chart_map.items():
                    ns = _norm_title(src_title)
                    if nt == ns or nt in ns or ns in nt:
                        chart = src_chart
                        break
            if chart is None:
                continue
            # 关键过滤：option 为 None 的图（未渲染出 option，仅有 raw_data 或无图）
            # 前端 EtherealRadarChart 等组件会对 chartNode.title 直接取值，null 会崩溃整页。
            # 与路由 report.py 的 not chart.get("option") 过滤保持一致。
            if not chart.get("option"):
                continue
            slot = chart.get("slot", "")
            # 同一 section 内按 slot 去重：LLM 重复声明同一张图（或 chart_titles
            # 含一个 slot 的多处引用）时，避免前端 key=slot 撞车（React 重复 key 警告）。
            if slot and slot in seen_slots:
                continue
            if slot:
                seen_slots.add(slot)
            bound.append({
                "title": chart.get("title", ""),
                "option": chart.get("option"),
                "chart_type": chart.get("chart_type", chart.get("type", "")),
                "slot": slot,
                "raw_data": chart.get("raw_data"),
                "role": chart.get("role", ""),
            })
        section["section_charts"] = bound

    return sections


def _build_fallback_from_packages(
    packages: List[Dict[str, Any]],
    report_input: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """AI 调用失败时，直接用 AnalysisPackage 中的已有数据构建报告"""
    sections: List[Dict[str, Any]] = []
    sections_data = report_input.get("sections_data", {})

    # 执行摘要
    total_kpis = 0
    total_insights = 0
    for pkgs in sections_data.values():
        for pkg in pkgs:
            total_kpis += len(pkg.get("kpis", []))
            total_insights += len(pkg.get("insights", [])) + len(pkg.get("conclusions", []))
    sections.append({
        "type": "executive_summary",
        "title": "执行摘要",
        "content": (
            f"本报告基于 {len(packages)} 个已完成的分析包生成。"
            f"共包含 {total_kpis} 项 KPI 指标和 {total_insights} 条数据洞察。"
        ),
    })

    # 数据概览
    data_profile = report_input.get("data_profile", {})
    if data_profile:
        time_cols = data_profile.get("time_cols", [])
        cat_cols = data_profile.get("category_cols", [])
        num_cols = data_profile.get("numeric_cols", [])
        sections.append({
            "type": "data_overview",
            "title": "数据概览",
            "content": (
                f"时间维度：{', '.join(time_cols) if time_cols else '无'}。"
                f"数值指标：{', '.join(num_cols) if num_cols else '无'}。"
                f"分类维度：{', '.join(cat_cols) if cat_cols else '无'}。"
            ),
        })
    else:
        sections.append({
            "type": "data_overview",
            "title": "数据概览",
            "content": f"基于 {len(packages)} 个分析包生成。分析包概要：{report_input.get('packages_summary', '')}",
        })

    # 各分析章节
    for section_name, pkgs in sections_data.items():
        insights = []
        for pkg in pkgs:
            question = pkg.get("business_question", "")
            conclusions = pkg.get("conclusions", [])
            pkg_insights = pkg.get("insights", [])
            kpis = pkg.get("kpis", [])

            analysis_parts = []
            if question:
                analysis_parts.append(f"业务问题：{question}")
            if kpis:
                kpi_texts = []
                for k in kpis[:5]:
                    cs = f" ({k.get('change', '')})" if k.get("change") else ""
                    kpi_texts.append(f"{k.get('label', '')}：{k.get('value', '')}{cs}")
                analysis_parts.append("；".join(kpi_texts))
            if conclusions:
                analysis_parts.extend(conclusions)
            if pkg_insights:
                analysis_parts.extend(pkg_insights[:3])

            chart_titles = []
            charts = pkg.get("chart_data", [])
            if charts:
                chart_titles = [c.get("title", "") for c in charts if c.get("title")]

            content = "。".join(analysis_parts) if analysis_parts else "暂无详细分析数据"

            sections.append({
                "type": section_name,
                "title": SECTION_DISPLAY_NAME.get(section_name, section_name),
                "content": content,
                "chart_titles": chart_titles,
            })

    # 管理建议
    all_conclusions = []
    for pkgs in sections_data.values():
        for pkg in pkgs:
            all_conclusions.extend(pkg.get("conclusions", []))
            all_conclusions.extend(pkg.get("recommendations", []))

    if all_conclusions:
        sections.append({
            "type": "management_suggestions",
            "title": "管理建议",
            "content": "；".join(str(c).strip() for c in all_conclusions[:5]) + "。",
            "chart_titles": [],
        })

    # 总结
    sections.append({
        "type": "conclusion",
        "title": "总结",
        "content": f"报告基于 {len(packages)} 个分析包自动生成。AI 报告生成失败，以上内容为已有分析数据的直接汇总。",
        "chart_titles": [],
    })

    return sections