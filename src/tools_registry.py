"""
七工具箱骨架（本次落地 clean_data，含 LLM 智能推荐）

每个工具都返回统一的 ToolResult 契约，供大脑方 Agent 后续串联成智能体工作流。
本次只实现 clean_data（数据清洗）一个工具，其余工具后续按同样契约补。

ToolResult 字段（以 TOOLS_CONTRACT 契约为准）：
- ok: bool                        工具是否成功执行
- data: Any                       成功时的返回数据
- error: str                      失败/跳过原因说明
- missing_columns: List[str]      数据不足时缺失的列名（如业务模型因缺列被跳过时收集所缺列）
- skipped_models: List[Dict[str, Any]]  因条件不足跳过的分析模型，每项 {model: str, missing_columns: List[str]}
- message: str                    给用户的建议文案
- （以下 3 个字段为契约外、仅供 clean_data 弹窗所需，保留不删）
- available_alternatives: List[Dict[str, Any]]  4 种可选填充方法（每项为 {method,label,description}）
- needs_orientation: bool         是否需要追问（数据/取向类）
- orientation_hint: str           追问提示语

clean_data 的设计（方案X + 兜底甲）：
- 体检态：先用确定性函数扫缺失值 + 类型问题，再调 Agnes 生成【一个全局缺失填充
  推荐策略 + 整体理由 + 4 种可选方法】；Agnes 失败则回退确定性默认推荐。
- 执行态：用户选 1 个 method，统一应用到所有缺失列；类型转换由 Agnes 自动推断
  完成（用户无感），Agnes 失败则回退确定性 detect_data_type_issues 自动转。
- 全程只产出新 df，绝不覆盖入参、不写 session。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import sys
import json
import logging

import pandas as pd

# 保证无论从哪个工作目录启动都能找到同目录下的 data_cleaner
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from data_cleaner import (
    get_missing_value_report,
    handle_missing_values,
    detect_data_type_issues,
    convert_column_type,
)

# 复用列映射模块的鲁棒 JSON 提取（支持代码块/截断/数组/尾部逗号），统一容错逻辑
from src.mapping.column_mapper import _robust_extract_json

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    ok: bool = True
    data: Any = None
    error: str = ""
    missing_columns: List[str] = field(default_factory=list)
    skipped_models: List[Dict[str, Any]] = field(default_factory=list)  # 每项: {"model": str, "missing_columns": List[str]}
    message: str = ""
    # 以下 3 个字段为契约外、仅供 clean_data 弹窗所需，保留不删
    available_alternatives: List[Dict[str, Any]] = field(default_factory=list)
    needs_orientation: bool = False
    orientation_hint: str = ""


# ---------------------------------------------------------------------------
# 清洗方法【写死清单】（必须来自 data_cleaner.handle_missing_values 的 method 枚举）
# 每个选项附优缺点，供大脑方转成前端选择框，AI 不得自创新方法。
# 注意：drop / drop_column（删行/删列）按用户明确要求已移出对外 options，
# 仅作为内部枚举保留，执行态不再接受。
# ---------------------------------------------------------------------------
MISSING_METHODS: Dict[str, Dict[str, Any]] = {
    "fill_mean": {
        "label": "均值填充",
        "description": "用该列的平均值填充缺失值（仅对数值列有意义）",
        "pros": ["保留样本量", "计算简单、稳定", "不引入新类别"],
        "cons": ["会改变数据分布（拉高集中度）", "对极端值敏感", "非数值列不可用"],
    },
    "fill_median": {
        "label": "中位数填充",
        "description": "用该列的中位数填充缺失值（仅对数值列有意义）",
        "pros": ["保留样本量", "对极端值比均值更稳健"],
        "cons": ["仍会改变分布", "非数值列不可用", "丢失缺失本身的信息"],
    },
    "fill_mode": {
        "label": "众数填充",
        "description": "用该列出现最多的值填充缺失值（分类/文本列友好）",
        "pros": ["适合类别型数据", "保留样本量", "不改变众数结构"],
        "cons": ["可能放大主导类别", "多众数时只取其一", "扭曲分布"],
    },
    "fill_0": {
        "label": "填 0",
        "description": "把所有缺失值填为 0（适用于缺失即“无”的计数/金额列）",
        "pros": ["语义清晰（缺失=没有）", "保留样本量", "计算简单"],
        "cons": ["会拉低均值", "不适合本就“缺失即无”语义的列", "扭曲分布"],
    },
    # 仅内部枚举，不进入对外 options
    "_drop": {
        "label": "删除缺失行",
        "description": "直接删掉该列含有缺失值的整行（已按需求移出选项）",
        "pros": [], "cons": [],
    },
    "_drop_column": {
        "label": "删除该列",
        "description": "整列缺失过多时直接丢弃该列（已按需求移出选项）",
        "pros": [], "cons": [],
    },
}

# 对外暴露给用户选择的 4 种方法（顺序即前端展示顺序）
USER_MISSING_METHODS: List[str] = [
    "fill_mean", "fill_median", "fill_mode", "fill_0",
]

# detect_data_type_issues 的“建议”中文 → target_type 映射
TYPE_ISSUE_SUGGESTION_MAP = {
    "转换为日期时间类型": "datetime",
    "转换为数值类型": "numeric",
    "转换为字符串类型": "string",
    "转换为分类类型": "category",
}

TYPE_OPTIONS_META: Dict[str, Dict[str, Any]] = {
    "datetime": {
        "label": "转换为日期时间",
        "description": "把该列解析为日期时间类型（可做趋势/时间分析）",
        "pros": ["解锁时间维度分析", "统一日期格式"],
        "cons": ["解析失败的脏数据会变成 NaT（又产生缺失）", "需列内容确为日期"],
    },
    "numeric": {
        "label": "转换为数值",
        "description": "把该列解析为数值类型（可做统计/聚合）",
        "pros": ["解锁数值统计与图表", "统一单位与精度"],
        "cons": ["非数字内容会变成 NaN（又产生缺失）", "需列内容确为数字"],
    },
    "string": {
        "label": "转换为字符串",
        "description": "把该列统一为字符串类型",
        "pros": ["避免类型混乱", "适合文本类列"],
        "cons": ["不能做数值运算", "占用空间略大"],
    },
    "category": {
        "label": "转换为分类",
        "description": "把该列设为分类类型（节省内存、便于分组）",
        "pros": ["分组/去重高效", "节省内存"],
        "cons": ["类别过多时意义不大", "需列内容确为有限类别"],
    },
}


# ---------------------------------------------------------------------------
# Agnes 结构化 JSON 调用（不复用 agent.analyze，独立走 JSON-only 协议）
# ---------------------------------------------------------------------------
def _get_agnes_client():
    """复用 agent.py 的 Agnes 客户端构造（model/base_url/key 读取逻辑）。

    返回 (client, model)。若环境无 AGNES_API_KEY，则抛错交由调用方捕获回退。
    """
    from src.ai_agent.agent import DataAnalysisAgent
    agent = DataAnalysisAgent()  # 内部读 os.environ["AGNES_API_KEY"]，空则抛 ValueError
    return agent.client, agent.model


def _call_agnes_json(system_prompt: str, user_prompt: str, timeout: float = 60.0) -> Dict[str, Any]:
    """调 Agnes 并要求只返回 JSON，解析失败抛异常（由调用方回退确定性规则）。

    不复用 analyze（analyze 被禁止结构化输出），这里独立发 chat.completions。
    """
    client, model = _get_agnes_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=2048,
        timeout=timeout,
    )
    text = (resp.choices[0].message.content or "").strip()
    # 复用鲁棒 JSON 提取：支持 ```json 代码块、前后解释文字、截断修复、数组、尾部逗号。
    # 解析失败抛 ValueError（含原文前 2000 字符），由调用方回退确定性规则。
    return json.loads(_robust_extract_json(text))


# ---------------------------------------------------------------------------
# 缺失值：确定性扫描 + LLM 全局推荐 + 兜底甲
# ---------------------------------------------------------------------------
_FIVE_METHODS_META = [
    {"id": "fill_mean", "method": "fill_mean", **MISSING_METHODS["fill_mean"]},
    {"id": "fill_median", "method": "fill_median", **MISSING_METHODS["fill_median"]},
    {"id": "fill_mode", "method": "fill_mode", **MISSING_METHODS["fill_mode"]},
    {"id": "fill_0", "method": "fill_0", **MISSING_METHODS["fill_0"]},
]


def _build_alternatives() -> List[Dict[str, Any]]:
    """4 种可选方法清单（来自 data_cleaner 写死方法，含 label/description/pros/cons）。"""
    return [
        {
            "id": m["id"],
            "label": m["label"],
            "method": m["method"],
            "description": m["description"],
            "pros": m["pros"],
            "cons": m["cons"],
        }
        for m in _FIVE_METHODS_META
    ]


def _scan_and_recommend(
    df: pd.DataFrame,
    missing_cols: List[str],
    missing_detail: Dict[str, int],
) -> Optional[Dict[str, Any]]:
    """体检态调 Agnes 生成【一个全局缺失填充推荐策略 + 整体理由 + 4 种备选】。

    成功返回 {"strategy": str, "reason": str, "alternatives": [...]}；
    任何异常（无 key / 超时 / 非 JSON / 字段缺失）→ 返回 None，由调用方走兜底甲。
    """
    try:
        # 构造给 LLM 的数据画像：列名、类型、缺失数、每列前 5 行样本（截断 token）
        col_profiles = []
        for col in missing_cols:
            dtype = str(df[col].dtype)
            sample = df[col].dropna().head(5).tolist()
            col_profiles.append({
                "column": col,
                "dtype": dtype,
                "missing_count": int(missing_detail.get(col, 0)),
                "missing_ratio": round(int(missing_detail.get(col, 0)) / max(len(df), 1) * 100, 1),
                "sample": [str(s) for s in sample],
            })
        data_json = json.dumps(col_profiles, ensure_ascii=False)

        system_prompt = (
            "你是资深数据清洗专家。用户希望对所有含缺失值的列【统一使用同一种填充方法】。"
            "请根据每列的数据特征（类型、缺失比例、样本值），从下面 4 种方法中选择【最合适的一种】"
            "作为全局统一策略，并给出整体理由。\n"
            "4 种方法（method 取值必须严格是以下之一）：\n"
            "1. fill_mean（均值填充，仅数值列）\n"
            "2. fill_median（中位数填充，仅数值列）\n"
            "3. fill_mode（众数填充，类别/文本列友好）\n"
            "4. fill_0（填 0，缺失即“无”的计数/金额列）\n"
            "你必须只返回一个 JSON 对象，格式严格如下，不要任何额外文字：\n"
            '{"recommended_method": "上述4种之一",'
            '"reason": "为什么所有列统一用这个方法的整体理由（中文，2-4句）",'
            '"alternatives": [4个对象，每个含 id/label/method/description/pros/cons，'
            "method 必须是上述4种之一且4个各不相同]}"
        )
        user_prompt = (
            f"数据集规模：{len(df)} 行 x {len(df.columns)} 列。\n"
            f"含缺失值的列画像（JSON）：\n{data_json}\n\n"
            "请输出全局统一填充策略的推荐 JSON。"
        )

        result = _call_agnes_json(system_prompt, user_prompt)
        # 容错：LLM 可能用不同 key 返回策略名
        strategy = (
            result.get("recommended_method")
            or result.get("method")
            or result.get("strategy")
            or result.get("fill_method")
        )
        strategy = (strategy or "").strip().lower() if strategy else None
        reason = result.get("reason") or result.get("rationale") or ""
        alts = result.get("alternatives") or []

        # 校验 strategy 合法性
        if strategy not in USER_MISSING_METHODS:
            return None
        # 校验 alternatives 完整性，不足 4 个则用写死清单补全。
        # 防御：LLM 可能返回 method 为非字符串（dict/list/None），
        # 直接用 `in` 会抛 TypeError(unhashable)，故先 isinstance 守门，
        # 否则整个 _scan_and_recommend 被 except 吞掉 → 错失兜底。
        valid_alts = [
            a for a in alts
            if isinstance(a.get("method"), str) and a.get("method") in USER_MISSING_METHODS
        ]
        if len(valid_alts) < 4:
            valid_alts = _build_alternatives()
        return {
            "strategy": strategy,
            "reason": reason,
            "alternatives": valid_alts,
        }
    except Exception as e:
        logger.warning(f"[clean_data] Agnes 缺失推荐失败，走兜底甲：{e}")
        return None


def _fallback_recommend(
    df: pd.DataFrame,
    missing_cols: List[str],
    missing_detail: Dict[str, int],
) -> Dict[str, Any]:
    """兜底甲：确定性默认推荐。

    规则：若所有缺失列都是数值列 → 中位数；若都是类别/文本列 → 众数；
    若混合 → 按多数列类型选一个主推策略，并在说明里逐列写明
    “数值列用中位数、文本列用众数”（本工具为全局统一填充，混合列需用户自行取舍）。
    """
    if not missing_cols:
        return {
            "strategy": "",
            "reason": "未发现缺失值，无需填充。",
            "alternatives": _build_alternatives(),
        }
    numeric_missing = [c for c in missing_cols if pd.api.types.is_numeric_dtype(df[c])]
    all_numeric = len(numeric_missing) == len(missing_cols) and len(missing_cols) > 0
    all_categorical = len(numeric_missing) == 0 and len(missing_cols) > 0

    if all_numeric:
        strategy = "fill_median"
        reason = "缺失列均为数值类型，建议用中位数填充：对极端值稳健、不改变分布形态，比均值更抗异常。"
    elif all_categorical:
        strategy = "fill_mode"
        reason = "缺失列均为类别/文本类型，建议用众数填充：保留主导类别结构、不改变数据语义。"
    else:
        # 混合类型：按多数列类型选主推，说明里逐列点明建议方法
        num_cols = [c for c in missing_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in missing_cols if not pd.api.types.is_numeric_dtype(df[c])]
        strategy = "fill_median" if len(num_cols) >= len(cat_cols) else "fill_mode"
        detail = "；".join(
            [f"{c}（数值）用中位数" for c in num_cols]
            + [f"{c}（文本/类别）用众数" for c in cat_cols]
        )
        reason = (
            "缺失列混合了数值与文本/类别类型，无法用单一方法完美统一填充。"
            f"按列类型建议：{detail}。本工具为全局统一填充，已为你选最多列适用的主推方法"
            f"（{MISSING_METHODS[strategy]['label']}），你也可按列自行取舍。"
        )

    return {
        "strategy": strategy,
        "reason": reason + "（注：本次为离线兜底推荐，未调用 AI）",
        "alternatives": _build_alternatives(),
    }


# ---------------------------------------------------------------------------
# 类型转换：LLM 自动推断 + 兜底（确定性 detect_data_type_issues）
# ---------------------------------------------------------------------------
def _infer_types_via_llm(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    """执行态调 Agnes 推断每列目标类型，返回 {列名: target_type}。

    失败（无 key / 超时 / 非 JSON）→ 返回 None，由调用方走确定性兜底。
    target_type 只能是 datetime/numeric/string/category 之一，其余丢弃。
    """
    try:
        col_profiles = []
        for col in df.columns:
            sample = df[col].dropna().head(5).tolist()
            col_profiles.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "sample": [str(s) for s in sample],
            })
        data_json = json.dumps(col_profiles, ensure_ascii=False)

        system_prompt = (
            "你是数据类型的识别专家。请根据列名与样本值，判断每列应当转换成的目标类型。"
            "目标类型只能从以下四选一：datetime（日期时间）、numeric（数值）、"
            "string（字符串）、category（有限类别）。\n"
            "只返回 JSON 对象，格式：{\"列名\": \"目标类型\", ...}，不要额外文字。"
            "若某列当前类型已合理、无需转换，则不出现在结果中。"
        )
        user_prompt = (
            f"数据集列画像（JSON）：\n{data_json}\n\n请输出类型转换映射 JSON。"
        )
        result = _call_agnes_json(system_prompt, user_prompt, timeout=60.0)
        if not isinstance(result, dict):
            return None
        valid = {"datetime", "numeric", "string", "category"}
        mapping = {k: v for k, v in result.items() if v in valid and k in df.columns}
        return mapping if mapping else None
    except Exception as e:
        logger.warning(f"[clean_data] Agnes 类型推断失败，走兜底：{e}")
        return None


def _fallback_types(df: pd.DataFrame) -> Dict[str, str]:
    """兜底：用确定性 detect_data_type_issues 推断类型映射。"""
    mapping: Dict[str, str] = {}
    for issue in detect_data_type_issues(df):
        col = issue.get("列名")
        suggestion_text = issue.get("建议", "")
        target = None
        for key, val in TYPE_ISSUE_SUGGESTION_MAP.items():
            if key in suggestion_text:
                target = val
                break
        if col and target:
            mapping[col] = target
    return mapping


def clean_data(df: pd.DataFrame, actions: Optional[List[Dict[str, Any]]] = None) -> ToolResult:
    """数据清洗工具（七工具箱之一），三态：

    1) 空 df：ok=False，reason 说明。
    2) 体检态（actions 为 None）：确定性扫描缺失值 + 类型问题，调 Agnes 生成
       【一个全局缺失填充推荐策略 + 整体理由 + 4 种备选】；Agnes 失败则兜底甲。
       类型问题也一并返回供参考（但类型转换在执行态由 LLM 自动完成，用户无感）。
    3) 执行态（actions 提供 method）：对所有缺失列统一应用该 method，
       并自动执行类型转换（LLM 推断，失败兜底确定性），返回新 df（不覆盖入参）。

    执行态 action 契约（方案X：全局统一，无 column 维度）：
      {"method": "fill_median"}            # 全局统一应用到所有缺失列

    类型转换不进用户选项，由工具在执行态内部自动完成。
    """
    # ---- 边界：空 df ----
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return ToolResult(ok=False, error="无数据：请先上传数据后再调用清洗工具")

    # ---- 执行态 ----
    if actions is not None:
        # 兼容两种调用：
        #   clean_data(df, "fill_median")            -> 字符串简写
        #   clean_data(df, [{"method": "fill_median"}]) -> 原契约(List[Dict])
        if isinstance(actions, str):
            method = actions
        elif isinstance(actions, list):
            if not actions:
                return ToolResult(ok=False, error="执行态未提供任何 action（需含 method）")
            first = actions[0]
            method = first.get("method") if isinstance(first, dict) else first
        else:
            return ToolResult(ok=False, error="执行态入参格式错误：method 应为字符串或 [{'method': ...}]")
        if method not in USER_MISSING_METHODS:
            return ToolResult(
                ok=False,
                error=f"不支持的填充方法：{method}。仅支持 {USER_MISSING_METHODS}",
            )

        df_new = df.copy()
        # 重新确定性扫描缺失列（不依赖入参传入 column，保证“所有缺失列统一处理”）
        missing_report = get_missing_value_report(df_new)
        missing_detail = missing_report.get("缺失详情", {}) or {}
        missing_cols = [c for c, n in missing_detail.items() if n and n > 0]

        applied = []
        failed = []
        for col in missing_cols:
            try:
                df_new = handle_missing_values(df_new, col, method)
                applied.append(f"{col} -> {method}")
            except Exception as e:  # 单步失败不中断其余
                failed.append(f"{col}: {str(e)}")

        # 类型转换：LLM 自动推断（失败兜底确定性）
        type_mapping = _infer_types_via_llm(df_new)
        type_source = "llm"
        if type_mapping is None:
            type_mapping = _fallback_types(df_new)
            type_source = "fallback"
        type_applied = []
        type_failed = []
        for col, target in type_mapping.items():
            try:
                df_new = convert_column_type(df_new, col, target)
                type_applied.append(f"{col} -> {target}")
            except Exception as e:
                type_failed.append(f"{col}: {str(e)}")

        summary = {
            "method": method,
            "missing_filled": applied,
            "missing_failed": failed,
            "types_converted": type_applied,
            "types_failed": type_failed,
            "type_infer_source": type_source,
            "original_shape": list(df.shape),
            "cleaned_shape": list(df_new.shape),
        }
        return ToolResult(
            ok=True,
            data={"cleaned_df": df_new, "summary": summary},
            message=(
                f"已对所有缺失列统一应用「{MISSING_METHODS[method]['label']}」，"
                f"成功 {len(applied)} 列、失败 {len(failed)} 列；"
                f"类型转换（{type_source}）成功 {len(type_applied)} 列、失败 {len(type_failed)} 列。"
            ),
        )

    # ---- 体检态 ----
    missing_report = get_missing_value_report(df)
    missing_detail = missing_report.get("缺失详情", {}) or {}
    missing_cols = [c for c, n in missing_detail.items() if n and n > 0]

    type_issues = detect_data_type_issues(df)

    # 调 Agnes 生成全局缺失填充推荐（失败 → 兜底甲）
    recommendation = _scan_and_recommend(df, missing_cols, missing_detail)
    rec_source = "llm"
    if recommendation is None:
        recommendation = _fallback_recommend(df, missing_cols, missing_detail)
        rec_source = "fallback"

    needs_cleaning = len(missing_cols) > 0 or len(type_issues) > 0
    profile = {
        "missing": {
            "missing_value_count": missing_report.get("缺失值列数", 0),
            "total_missing": missing_report.get("总缺失值数", 0),
            "columns": {c: int(n) for c, n in missing_detail.items() if n and n > 0},
        },
        "type_issues": [
            {"column": i.get("列名"), "problem": i.get("问题"), "suggestion": i.get("建议")}
            for i in type_issues
        ],
    }

    suggestion = "无需清洗" if not needs_cleaning else \
        f"发现 {len(missing_cols)} 列缺失值、{len(type_issues)} 列类型问题。" + \
        (f"（缺失填充推荐由 AI 给出）" if rec_source == "llm" else "（AI 不可用，已用离线兜底推荐）")

    return ToolResult(
        ok=True,
        data={
            "needs_cleaning": needs_cleaning,
            "profile": profile,
            "recommendation": {
                "strategy": recommendation["strategy"],
                "reason": recommendation["reason"],
                "alternatives": recommendation["alternatives"],
            },
            "recommendation_source": rec_source,
        },
        missing_columns=missing_cols,
        available_alternatives=recommendation["alternatives"],
        message=suggestion,
    )


# ---------------------------------------------------------------------------
# Chat 智能体专用工具（df 由 agent._resolve_tool_call 注入，不直接走 dispatch）
# ---------------------------------------------------------------------------

def profile_data(df: pd.DataFrame) -> ToolResult:
    """返回当前 df 的数据侦察快照（列名/类型/缺失值/数值统计/行数）。

    实际侦察走 data_recon.scan；若会话已存 snapshot 可直接复用，但本函数
    始终对传入 df 实时扫描，保证与当前 session 数据一致。
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return ToolResult(ok=False, error="无数据：请先上传数据后再调用该工具")
    from data_recon import scan
    snapshot = scan(df)
    return ToolResult(
        ok=True,
        data={"profile": snapshot},
        message=f"共 {snapshot['rows']} 行 × {snapshot['column_count']} 列，"
                   f"{snapshot['missing_overview']['cols_with_missing']} 列存在缺失值。",
    )


def run_template(df: pd.DataFrame, intents: Optional[List[str]] = None,
                  manager=None, session_id: Optional[str] = None) -> ToolResult:
    """运行后端分析引擎匹配到的分析模型，返回 AnalysisPackage 摘要，并把完整包写入 session。

    注意：列名映射(required_columns 匹配)由调用方在注入 df 前完成，本函数
    仅负责跑 run_analysis。

    当传入 manager / session_id 时，把每个包的完整 dict 写入
    session.analysis_packages（key=pkg_id），供 generate_chart / build_dashboard /
    generate_report 产出工具读取（思路甲：分析输出存 ctx，LLM 不手传大 JSON）。
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return ToolResult(ok=False, error="无数据：请先上传数据后再调用分析工具")
    try:
        from analysis_engine.engine import run_analysis
        from analysis_engine.registry import get_models
        packages = run_analysis(df, intents)
    except Exception as e:
        return ToolResult(ok=False, error=f"分析执行异常：{str(e)}")

    # ⚠️契约缺口补全：业务模型分析会遍历模型、对缺列模型静默跳过，
    # 此处确定性比对 registry，把被跳过模型与其所缺列回填 skipped_models / missing_columns，
    # 让大脑方/前端可知「为什么某些模型没跑」。intents 非空时只判定指定模型，避免无关模型假阳性。
    skipped_models: List[Dict[str, Any]] = []
    missing_set: set = set()
    try:
        intent_set = set(intents or [])
        for m in get_models():
            if intent_set and m.name not in intent_set:
                continue
            if not m.can_run(df):
                miss = [c for c in (m.required_columns or []) if c not in set(df.columns)]
                skipped_models.append({"model": m.name, "missing_columns": miss})
                missing_set.update(miss)
    except Exception:
        pass
    missing_columns = sorted(missing_set)

    if not packages:
        return ToolResult(
            ok=True,
            data={"packages": [], "full_packages": []},
            message="未匹配到任何分析模型（当前数据列名可能不满足模型的 required_columns）。"
                      "可尝试先清洗数据，或用 run_python 做自定义分析。",
            skipped_models=skipped_models,
            missing_columns=missing_columns,
        )

    # 完整包 dict 列表：对象调 .to_api_dict()（AnalysisPackage 的序列化方法），
    # 已是 dict 则直接用。注意：AnalysisPackage 无 to_dict()，只有 to_api_dict()。
    full_packages: List[Dict[str, Any]] = []
    for pkg in packages:
        try:
            if hasattr(pkg, "to_api_dict"):
                full_packages.append(pkg.to_api_dict())
            elif isinstance(pkg, dict):
                full_packages.append(pkg)
        except Exception:
            continue

    # 写入 session.analysis_packages（供产出工具消费）；兼容 value 已是 dict 的情况。
    if manager is not None and session_id is not None and full_packages:
        try:
            session = manager.get_session(session_id)
            if session is not None:
                existing = getattr(session, "analysis_packages", None)
                if not isinstance(existing, dict):
                    existing = {}
                for pkg in full_packages:
                    pkg_id = str(pkg.get("id") or pkg.get("analysis_type") or f"pkg_{len(existing)}")
                    existing[pkg_id] = pkg
                manager.set_analysis_packages(session_id, existing)
        except Exception as e:
            logger.warning("写入 session.analysis_packages 失败：%s", e)

    # 摘要：每个包给出类型 + 一句结论，供 LLM 组织语言回用户
    summaries = []
    for pkg in full_packages:
        try:
            title = pkg.get("analysis_type") or pkg.get("title") or "分析"
            conclusion = pkg.get("conclusion") or ""
            summaries.append({"type": str(title), "conclusion": str(conclusion)[:500]})
        except Exception:
            summaries.append({"type": "分析", "conclusion": ""})

    return ToolResult(
        ok=True,
        data={"packages": summaries, "full_packages": full_packages, "package_count": len(summaries)},
        message=f"已运行 {len(summaries)} 个分析模型。",
        skipped_models=skipped_models,
        missing_columns=missing_columns,
    )


def run_python(df: pd.DataFrame, code: str) -> ToolResult:
    """执行 LLM 生成的 Python 代码做自定义分析（代码中可用 df/pd/np）。

    实际执行走 agent._execute_code；为避免循环依赖，这里延迟导入 agent 模块。
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return ToolResult(ok=False, error="无数据：请先上传数据后再调用该工具")
    if not code or not code.strip():
        return ToolResult(ok=False, error="未提供要执行的代码")
    try:
        from ai_agent.agent import _execute_code_structured
        result = _execute_code_structured(code, df, timeout_sec=18)
    except Exception as e:
        return ToolResult(ok=False, error=f"代码执行异常：{str(e)}")

    if not isinstance(result, dict):
        result = {"text": str(result), "chart": None}
    if "error" in result:
        return ToolResult(
            ok=False,
            error=result.get("error", "代码执行失败。"),
            message="请检查代码后重试，避免危险调用或白名单外的导入。",
        )

    text = result.get("text", "") or "代码执行成功，但无结论文本。"
    chart = result.get("chart") or None

    # 对齐 run_template 的 data 结构（packages 摘要），并把 chart 提升到顶层供前端直接取
    data = {
        "packages": [{
            "type": "custom_python",
            "conclusion": text,
            "chart": chart,
        }],
        "package_count": 1,
        "chart": chart,  # 顶层提升，前端从 tr.data.chart 取（不能用 tr.chart）
    }
    return ToolResult(
        ok=True,
        data=data,
        message="代码已执行，结论见 packages[0].conclusion。",
    )


def run_analysis(df: pd.DataFrame) -> ToolResult:
    """通用统计分析：自动识别数值列并返回每列的 7 项基础统计指标。

    LLM 无需指定哪些列要统计——工具内部通过 identify_fields 自动识别数值指标列，
    然后对每一列计算：total/mean/median/max/min/std/count。
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return ToolResult(ok=False, error="无数据：请先上传数据后再调用该工具")
    try:
        from report_analyzer import identify_fields, compute_basic_stats
        fields = identify_fields(df)
        metrics = fields.get("metrics", [])
        if not metrics:
            return ToolResult(
                ok=True,
                data={"basic_stats": {}, "columns_summary": {"numeric_count": 0, "numeric_columns": []}},
                message="当前数据中未识别到数值列，无法做基础统计。",
            )
        stats = compute_basic_stats(df, metrics)
        summary_lines = []
        for col, s in stats.items():
            parts = [f"{col}: 总计={s['total']}, 均值={s['mean']}, 中位数={s['median']}, "
                     f"最大={s['max']}, 最小={s['min']}, 标准差={s['std']}, 有效值={s['count']}条"]
            summary_lines.append(" ".join(parts))
        suggestion = "\n".join(summary_lines) if summary_lines else "统计完成"
        return ToolResult(
            ok=True,
            data={
                "basic_stats": stats,
                "columns_summary": {"numeric_count": len(metrics), "numeric_columns": metrics},
            },
            message=suggestion,
        )
    except Exception as e:
        return ToolResult(ok=False, error=f"通用统计执行异常：{str(e)}")


def generate_chart(df: pd.DataFrame, chart_type: str, **kwargs) -> ToolResult:
    """图表生成工具：根据指定图表类型和参数生成 ECharts option。

    支持两种数据传入方式（任选其一）：
    1. 传【数据框列名】：x="流量来源", y="订单实付金额"（引擎自动 groupby 聚合）；
    2. 传【现成数组字面量】：x='["天猫","私域"]', y='[731,730]' 或
       data='[{"维度":"口红","数值":169}, ...]'（直接作图，无需再聚合）。
    工具底层调用 echart_generator.create_chart 生成前端可渲染的 ECharts option。
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return ToolResult(ok=False, error="无数据：请先上传数据后再调用该工具")
    if not chart_type or not chart_type.strip():
        return ToolResult(ok=False, error="未指定图表类型（chart_type）")
    try:
        from echart_generator import create_chart, CHART_FUNCTIONS
        chart_type = chart_type.strip().lower()
        if chart_type not in CHART_FUNCTIONS:
            supported = sorted(CHART_FUNCTIONS.keys())
            logger.warning(
                "generate_chart 不支持的类型: %s | df列: %s | kwargs: %s",
                chart_type, list(df.columns), kwargs,
            )
            return ToolResult(
                ok=False,
                error=f"不支持的图表类型：{chart_type}。可选：{', '.join(supported)}",
            )
        option = create_chart(df, chart_type, **kwargs)
        if option is None:
            logger.warning(
                "generate_chart 返回None: chart_type=%s kwargs=%s df列=%s",
                chart_type, kwargs, list(df.columns),
            )
            return ToolResult(
                ok=False,
                error=f"图表类型 '{chart_type}' 在当前数据上无法生成（数据列不匹配或数据量不足）。",
            )
        # 与 run_python 保持一致：data.chart 顶层提升 + data.packages[0].chart 嵌套
        data = {
            "packages": [{
                "type": chart_type,
                "conclusion": kwargs.get("title", f"{chart_type} 图表"),
                "chart": option,
            }],
            "package_count": 1,
            "chart": option,
        }
        return ToolResult(
            ok=True,
            data=data,
            message=f"已生成 {chart_type} 图表。",
        )
    except Exception as e:
        logger.exception("generate_chart 异常 chart_type=%s kwargs=%s df列=%s",
                         chart_type, kwargs, list(df.columns) if df is not None else None)
        return ToolResult(ok=False, error=f"图表生成异常：{str(e)}")


def _collect_analysis_packages(manager, session_id: str) -> List[Dict[str, Any]]:
    """从 session.analysis_packages 收集完整 AnalysisPackage dict 列表，供产出工具消费。

    兼容 value 已是 dict（重启后落库反序列化）或仍是 AnalysisPackage 对象（调 .to_dict()）两种情况。
    """
    try:
        session = manager.get_session(session_id)
        if session is None:
            return []
        raw = getattr(session, "analysis_packages", None)
        if not isinstance(raw, dict) or not raw:
            return []
        packages: List[Dict[str, Any]] = []
        for val in raw.values():
            if hasattr(val, "to_api_dict"):
                try:
                    packages.append(val.to_api_dict())
                except Exception:
                    continue
            elif isinstance(val, dict):
                packages.append(val)
        return packages
    except Exception as e:
        logger.warning("读取 session.analysis_packages 失败：%s", e)
        return []


def generate_report(manager, session_id: str) -> ToolResult:
    """生成数据分析报告：读取 session.analysis_packages 的完整分析包，调 ReportBuilder 生成结构化报告。

    适合在用户表达「生成报告」「分析报告」等意图时由 LLM 调用。报告基于前面三分析
    （业务模型分析/通用统计/自由写码）写入 session 的完整分析包，不重新跑分析。
    """
    packages = _collect_analysis_packages(manager, session_id)
    if not packages:
        return ToolResult(
            ok=False,
            error="暂无分析数据：请先调用业务模型分析/通用统计/自由写码等分析工具，"
                   "待分析结果写入后我再生成报告。",
            message="请先让我对数据做分析（例如「分析一下这份数据的趋势和排行」），再要报告。",
        )
    try:
        # generate_report_from_packages 在 agent.py 内，延迟导入避免循环依赖。
        from ai_agent.agent import DataAnalysisAgent
        # generate_report_from_packages 是 DataAnalysisAgent 实例方法。缓存单例，避免每次调用重复构造。
        # 构造仅读环境变量 AGNES_API_KEY，不触发网络；报告内的 LLM 调用在方法内部按需发生。
        _agent = getattr(generate_report, "_agent_cache", None)
        if _agent is None:
            _agent = DataAnalysisAgent()
            generate_report._agent_cache = _agent
        result = _agent.generate_report_from_packages(packages, data_profile=None)
        sections = result.get("sections") or []
        title = result.get("report_title") or "数据分析报告"
        # sections 内每个 section 已通过 _bind_package_charts_to_sections 绑定 section_charts
        # （含 option/chart_type/title/raw_data），前端从 section.section_charts 取图，无需顶层 charts。
        return ToolResult(
            ok=True,
            data={
                "report": {
                    "report_title": title,
                    "sections": sections,
                    "packages_used": result.get("packages_used", 0),
                    "degradation": result.get("degradation", False),
                    "warning": result.get("warning"),
                }
            },
            message=f"已生成报告《{title}》，共 {len(sections)} 个章节。",
        )
    except Exception as e:
        logger.exception("generate_report 异常")
        return ToolResult(ok=False, error=f"报告生成异常：{str(e)}")


def _narrate_widgets(widget_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """确定性叙事编排（无 LLM）：为 chat 大屏计算 ABC 关联并就地写回。

    - A 同主题递进 (progressive)：相同 business_topic 的 ≥2 个 widget 归为一组叙事段落。
    - B 因果/时间对照 (contrast)：相同 business_topic 且 chart_type 不同（同主题用不同图对比），
      或 analysis_type 语义序相邻（如 trend/growth 配 comparison/concentration）视为对照。
    - C 系统相关 (related)：相同 analysis_type 但不同 business_topic 的 widget 自动归组。
    - 落单 widget → solo。

    就地修改 widget_dicts 的每个元素（追加 narrative），并返回 layout 字典。
    layout.order / 每个 block.widget_ids 一律取 widget 自身已有的 id（兜底 w{i}）。
    """
    # 兜底补 id，保证 order 完整
    for i, d in enumerate(widget_dicts):
        if not d.get("id"):
            d["id"] = f"w{i}"

    # ---- 按 business_topic 归组（A / B 的载体）----
    topic_groups: Dict[str, List[Dict[str, Any]]] = {}
    for d in widget_dicts:
        topic = d.get("business_topic") or "未分类"
        topic_groups.setdefault(topic, []).append(d)

    # ---- 按 analysis_type 归组（C 的载体）----
    type_groups: Dict[str, List[Dict[str, Any]]] = {}
    for d in widget_dicts:
        atype = d.get("analysis_type") or "unknown"
        type_groups.setdefault(atype, []).append(d)

    # 语义序相邻表（时间/因果相邻的分析类型，用于 B 类判定）
    SEQUENCE_NEIGHBORS = {
        "growth": {"trend", "comparison", "concentration"},
        "trend": {"growth", "comparison", "concentration", "anomaly"},
        "comparison": {"growth", "trend", "concentration", "ranking"},
        "concentration": {"growth", "trend", "comparison"},
        "ranking": {"comparison", "structure", "proportion"},
        "structure": {"proportion", "ranking"},
        "proportion": {"structure", "ranking"},
        "correlation": {"distribution"},
        "distribution": {"correlation"},
        "anomaly": {"trend", "comparison"},
        "funnel": {"retention", "conversion"},
        "retention": {"funnel", "conversion"},
        "geo": set(),
    }

    blocks: List[Dict[str, Any]] = []
    block_counter = 0
    # 记录每个 widget 的最终 relation_type（避免重复赋值冲突）
    assigned: Dict[str, str] = {}

    # 先处理 A/B：同 business_topic 的多 widget 组
    for topic, members in topic_groups.items():
        if len(members) >= 2:
            block_counter += 1
            block_id = f"block_{block_counter}"
            chart_types = {m.get("chart_type") for m in members}
            # B 类：同主题且图表类型多样 → 对照；否则 A 类递进
            if len(chart_types) >= 2:
                relation = "contrast"
                block_title = f"{topic}·对照"
            else:
                relation = "progressive"
                block_title = f"{topic}·叙事"
            widget_ids = [m["id"] for m in members]
            for m in members:
                assigned[m["id"]] = relation
            blocks.append({
                "block_id": block_id,
                "title": block_title,
                "widget_ids": widget_ids,
                "relation_type": relation,
            })

    # 再处理 C：相同 analysis_type 但跨不同 business_topic 的归组
    for atype, members in type_groups.items():
        distinct_topics = {m.get("business_topic") for m in members}
        if len(members) >= 2 and len(distinct_topics) >= 2:
            block_counter += 1
            block_id = f"block_{block_counter}"
            widget_ids = [m["id"] for m in members]
            for m in members:
                # 仅在尚未被 A/B 赋值时标 related，避免覆盖对照关系
                if m["id"] not in assigned:
                    assigned[m["id"]] = "related"
            # 已被 A/B 占用的成员仍纳入 block 展示，但 relation 取 related 作 block 级
            blocks.append({
                "block_id": block_id,
                "title": f"{atype}·相关",
                "widget_ids": widget_ids,
                "relation_type": "related",
            })

    # 兜底：未分配的 widget 标 solo
    for d in widget_dicts:
        rid = d["id"]
        if rid not in assigned:
            assigned[rid] = "solo"

    # 写回每个 widget 的 narrative（就地修改同一个 dict）
    for d in widget_dicts:
        rid = assigned[d["id"]]
        # 找到所属 block 标题
        blk_title = ""
        blk_id = ""
        for b in blocks:
            if d["id"] in b["widget_ids"]:
                blk_title = b["title"]
                blk_id = b["block_id"]
                break
        d["narrative"] = {
            "block_id": blk_id,
            "block_title": blk_title,
            "relation_type": rid,
        }

    # order：保持当前（已按 importance_score 降序）顺序
    order = [d["id"] for d in widget_dicts]

    return {"blocks": blocks, "order": order}


def build_dashboard(manager, session_id: str) -> ToolResult:
    """生成数据大屏：读取 session.analysis_packages 的完整分析包，调 widget_generator 生成 Widget 列表预览。

    适合在用户表达「生成大屏」「可视化大屏」「驾驶舱」等意图时由 LLM 调用。大屏是确定性
    转换（无 LLM、无重算），直接把分析包渲染为大屏 Widget 网格。
    """
    packages = _collect_analysis_packages(manager, session_id)
    if not packages:
        return ToolResult(
            ok=False,
            error="暂无分析数据：请先调用业务模型分析/通用统计/自由写码等分析工具，"
                   "待分析结果写入后我再生成大屏。",
            message="请先让我对数据做分析（例如「分析一下这份数据的结构」），再要大屏。",
        )
    try:
        from src.dashboard.widget_generator import WidgetGenerator
        widgets = WidgetGenerator().generate_from_dicts(packages)
        # 按优先级排序后序列化为可渲染的 dict 列表（兼容重启后 value 已是 dict）。
        widget_dicts = []
        for w in widgets:
            if hasattr(w, "to_dict"):
                widget_dicts.append(w.to_dict())
            elif isinstance(w, dict):
                widget_dicts.append(w)
        widget_dicts.sort(key=lambda d: d.get("importance_score", 0) or 0, reverse=True)
        # 叙事编排：就地写回每个 widget 的 narrative，并返回 layout
        layout = _narrate_widgets(widget_dicts)
        return ToolResult(
            ok=True,
            data={
                "bigscreen": {
                    "widgets": widget_dicts,
                    "widget_count": len(widget_dicts),
                    "layout": layout,
                }
            },
            message=f"已生成数据大屏，共 {len(widget_dicts)} 个组件。",
        )
    except Exception as e:
        logger.exception("build_dashboard 异常")
        return ToolResult(ok=False, error=f"大屏生成异常：{str(e)}")


# ---------------------------------------------------------------------------
# 工具分发器（大脑方后续在此注册更多工具）
# ---------------------------------------------------------------------------
_TOOLS = {
    "clean_data": clean_data,
    "profile_data": profile_data,
    "run_template": run_template,
    "run_analysis": run_analysis,
    "generate_chart": generate_chart,
    "run_python": run_python,
    "generate_report": generate_report,
    "build_dashboard": build_dashboard,
}


def get_function_definitions() -> List[Dict[str, Any]]:
    """返回 OpenAI function calling 格式的工具定义（传给 LLM 的候选全集）。

    注意：必须是标准 OpenAI tools 格式——外层 {"type": "function", "function": {...}}，
    否则 Agnes 网关会静默忽略整个 tools 参数，导致 LLM 永不触发 tool_calls。

    实际每轮下发的工具子集由 agent.py 的状态机（_is_dataset_cleaned / _has_clean_intent /
    _has_chart_intent / _has_report_intent / _has_bigscreen_intent / _has_analysis_intent）
    筛选：profile_data 仅供上传时后端自动侦察使用，分析对话中不再下发给 LLM；其余工具
    （clean_data / run_template / run_analysis / run_python /
    generate_chart / generate_report / build_dashboard）均由 LLM 按用户意图自行调用编排，
    其中三分析工具会把完整分析包写入会话，产出工具（图/大屏/报告）读取它作为输入。
    """
    from echart_generator import CHART_FUNCTIONS  # 延迟导入，避免循环依赖
    return [
        {
            "type": "function",
            "function": {
                "name": "profile_data",
                "description": "获取当前数据的结构快照（列名、类型、缺失值、基本统计、行数）。"
                               "仅在确实未掌握数据列名/类型时调用；数据快照通常已附在上下文中，"
                               "不要重复调用。",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "clean_data",
                "description": "数据清洗工具。调用时【不要传 method】：工具会扫描缺失值并返回可选的"
                               "填充方案（含推荐做法说明），由系统以弹窗形式交给用户选择；"
                               "用户选定后系统会自动续接并真正执行清洗。你只需在分析到需要清洗时"
                               "不带 method 地调用本工具，不要在用户选择前自行决定填充方式。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["fill_mean", "fill_median", "fill_mode", "fill_0"],
                            "description": "清洗方法，仅当用户已从弹窗选择后续接时才由系统传入；"
                                           "你正常调用时不要填此参数。fill_mean=均值, fill_median=中位数, "
                                           "fill_mode=众数, fill_0=填0",
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_template",
                "description": "运行后端已有的分析模型（RFM/留存/漏斗/聚类/关联规则/趋势等），"
                               "返回匹配到的模型结论摘要。intents 为空数组时自动匹配所有可用模型。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "intents": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "要运行的分析意图列表，空数组表示自动匹配所有可用模型",
                        }
                    },
                    "required": ["intents"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": "执行 Python 代码进行自定义数据分析。代码中可用 df（当前数据）、"
                               "pd（pandas）、np（numpy）。仅用于分析，不要做破坏性操作。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "要执行的 Python 代码"}
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_analysis",
                "description": "通用统计分析：自动识别数据中的数值列，对每列计算总和/均值/中位数/"
                               "最大值/最小值/标准差/有效值数量共 7 项基础指标。无需传参，工具自动完成。"
                               "适合在用户问「这个数据的整体情况」「均值方差」等基础统计问题时调用。",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_chart",
                "description": "根据指定图表类型和数据列生成 ECharts 图表。支持 30 种图表类型，"
                               "包括 bar/line/pie/scatter/heatmap/radar/treemap 等。"
                               "需要传入 chart_type（必填）以及对应的数据列参数（如 x/y/title）。"
                               "适合在用户要求「画个图」「做一个柱状图」等可视化需求时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chart_type": {
                            "type": "string",
                            "description": "图表类型，必填",
                            "enum": sorted(CHART_FUNCTIONS.keys()),
                        },
                        "x": {
                            "type": "string",
                            "description": "X 轴数据。可传【数据框列名】（如 \"流量来源\"）或【现成数组字面量】（如 '[\"天猫\",\"私域\"]'）。与 y 配合使用。",
                        },
                        "y": {
                            "type": "string",
                            "description": "Y 轴数据。可传【数据框列名】（如 \"订单实付金额\"）或【现成数组字面量】（如 '[731,730]'）。与 x 配合使用。",
                        },
                        "data": {
                            "type": "string",
                            "description": "现成数据数组（JSON 字符串），形如 '[{\"维度\":\"口红\",\"数值\":169}, ...]'。当不便给 x/y 列名时，可直接传入此聚合结果。",
                        },
                        "title": {
                            "type": "string",
                            "description": "图表标题",
                        },
                    },
                    "required": ["chart_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_report",
                "description": "生成数据分析报告：基于前面已运行的分析（业务模型分析/通用统计/自由写码）"
                               "写入会话的分析结果，产出结构化的文字报告（含各章节结论与配套图表）。"
                               "无需传参，工具自动读取会话中的分析输出。适合在用户要求「生成报告」「分析报告」"
                               "等意图时调用。注意：调用本工具前，请确保已先跑过分析工具。",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "build_dashboard",
                "description": "生成数据大屏：基于前面已运行的分析（业务模型分析/通用统计/自由写码）"
                               "写入会话的分析结果，把分析结论渲染为数据大屏组件预览（图表/KPI/表格网格）。"
                               "无需传参，工具自动读取会话中的分析输出。适合在用户要求「生成大屏」「可视化大屏」"
                               "「驾驶舱」等意图时调用。注意：调用本工具前，请确保已先跑过分析工具。",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]


def get_tool(name: str, args: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> ToolResult:
    """按名称分发到对应工具（契约接口名 get_tool）。"""
    tool = _TOOLS.get(name)
    if tool is None:
        return ToolResult(ok=False, error=f"未注册的工具：{name}")
    try:
        return tool(**(args or {}))
    except TypeError as e:
        return ToolResult(ok=False, error=f"工具 {name} 参数错误：{str(e)}")
    except Exception as e:
        return ToolResult(ok=False, error=f"工具 {name} 执行异常：{str(e)}")


# 过渡别名：保留以避免其它地方漏改导致调用失败（grep 确认无真实调用后可删）
def dispatch(*args, **kwargs):
    return get_tool(*args, **kwargs)


def list_tools() -> List[str]:
    return list(_TOOLS.keys())
