"""
ECharts 图表生成模块 - 输出 ECharts option JSON
支持与 Plotly chart_generator 相同的图表类型 + ECharts 独有的 brush/timeline 等交互
"""
import json
import math
import re
from collections import deque
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

# ========== 降采样护栏常量 ==========
_MAX_SERIES_POINTS = 2000   # 线/散/柱/面积单序列点上限
_MAX_CATEGORY = 50          # 类目轴类别上限
_MAX_PIE_SLICES = 20        # 饼图/树图/词云扇区上限
_COHORT_WINDOW_MONTHS = 12   # cohort 图表滚动窗口大小（首单月行数 × Index_j 列数）

# ========== 字面量兼容层占位列名 ==========
# LLM 传数组字面量（如 x='["天猫","私域"]'）时，_coerce_literal_df 会构造
# 内部 DataFrame，列名固定为 __x__/__y__。这些内部占位名绝不能当作轴名/系列名
# 写进 option，否则前端会渲染出 "y/x" 挤在轴外的乱码（见 create_chart 字面量路径）。
_PLACEHOLDER_COLS = {"__x__", "__y__"}


def _is_placeholder_col(name: Optional[str]) -> bool:
    """判断列名是否为字面量兼容层的内部占位列名（__x__/__y__）。"""
    return name in _PLACEHOLDER_COLS


def _axis_name_or_none(name: Optional[str]) -> Optional[str]:
    """返回可安全写入 xAxis.name/yAxis.name/series.name 的轴名：
    内部占位列名（__x__/__y__）返回 None（不渲染轴名），其余原样返回。"""
    if _is_placeholder_col(name):
        return None
    return name

# ★ Galaxy AI Analytics 统一配色（与前端 frontend/src/theme 模块保持一致）
# 10 色有序分类色板，禁止彩虹 / 每图随机配色。后端无法 import TS，此为常数镜像，
# 修改颜色时务必同步 frontend/src/theme/Palette.ts 与 ChartStyle.ts。
# BLUE_PALETTE 必须与前端 ChartStyle.series 完全一致（蓝→靛→青→金→粉→橙→青柠→淡紫→天空蓝→湖绿，暖色前置，AI 紫禁入图表）。
BLUE_PALETTE = [
    "#38BDF8", "#818CF8", "#22D3EE", "#FBBF24",
    "#F472B6", "#FB923C", "#84CC16", "#C084FC",
    "#60A5FA", "#2DD4BF",
]
# 向后兼容别名（Chart Factory 内统一使用 BLUE_PALETTE）
WARM_COLORS = BLUE_PALETTE

# Galaxy 主题常量（Single Source of Truth 的 Python 镜像）
GALAXY = {
    "page_bg": "#020617",
    "card_bg": "#0F172A",
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "text_muted": "#94A3B8",
    "axis": "rgba(248,250,252,0.55)",
    "primary": "#38BDF8",
    "primary_hover": "#7DD3FC",
    "primary_active": "#0EA5E9",
    "primary_bright": "#67E8F9",
    "sky": "#0ea5e9",
    "sky_mid": "#0369a1",
    "sky_deep": "#0c4a6e",
    "ai": "#8B5CF6",
    "interaction": "#22D3EE",
    "map_normal": "#23304E",
    "heat_start": "#13243F",
    "success": "#34D399",
    "danger": "#FB7185",
    "warning": "#FBBF24",
    "grid": "rgba(255,255,255,0.08)",
    "border": "rgba(255,255,255,0.08)",
    "tooltip_bg": "#0F172A",
    "tooltip_border": "rgba(255,255,255,0.08)",
    "tooltip_content": "#CBD5E1",
}

# ECharts 深色主题基础配置（统一 Galaxy 蓝）
DARK_THEME = {
    "backgroundColor": "transparent",
    "textStyle": {"color": GALAXY["text_secondary"]},
    "title": {"textStyle": {"color": GALAXY["text_primary"]}},
    "tooltip": {
        "backgroundColor": GALAXY["tooltip_bg"],
        "borderColor": GALAXY["tooltip_border"],
        "textStyle": {"color": GALAXY["text_primary"]}
    },
    "legend": {
        "textStyle": {"color": GALAXY["text_secondary"]},
        "top": "bottom"
    },
    "toolbox": {
        "feature": {
            "saveAsImage": {"title": "下载为PNG", "backgroundColor": "transparent"},
            "dataView": {"title": "数据视图", "readOnly": True},
        }
    },
}


def _get_default_title(title: str) -> dict:
    return {"text": title, "left": "center", "top": 8, "textStyle": {"color": GALAXY["text_primary"], "fontSize": 14}}


def _interval_left(v) -> float:
    """提取区间字符串 '(a, b]' / '[a, b)' 的左端点数值；非区间返回 inf 排末尾。"""
    if not isinstance(v, str):
        return float("inf")
    m = re.match(r"^[\(\[]\s*([-+]?\d*\.?\d+)", v)
    return float(m.group(1)) if m else float("inf")


def _numeric_key(v) -> float:
    """把分桶标签解析为可排序的数值键。

    - 纯数字字符串 "0"/"15"/"53" → 该数字
    - 尾桶 "≥157天"/"≥53天" → 该数字（保证排在普通区间之后）
    - 区间字符串 "(a, b]"/"[a, b)" → 左端点（兜底，主路径已由 _interval_left 处理）
    - 无法解析 → +inf（排末尾，避免污染普通文本列的字典序）
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float("inf")
    s = str(v).strip()
    if s == "":
        return float("inf")
    # 容忍 "≥157天" / ">=157天" / "(a, b]" 等前缀，提取首个数字
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if m:
        return float(m.group(0))
    return float("inf")


def _sort_data(df, x, y):
    """按 x 排序数据。

    - 普通数值/分类列：沿用 sort_values 原行为。
    - 区间字符串列（形如 '(a, b]'）：按左端点数值排序，确保直方图 X 轴随数值递增
      （qcut 分位数分箱区间宽度差异大，字典序排序会乱序，长尾形态不可见）。
      仅当列样本命中区间格式才启用，普通分类列（省份/类目等）不受影响。
    - 纯数字分桶标签（如 churn_rule 的 "0"/"15"/"53" 与尾桶 "≥157天"）：
      按数值排序，避免字符串字典序 "0"<"106"<"15" 导致的 X 轴乱序；
      尾桶因数值最大（≥N）自然落在最右。普通文本列（省份/类目等）数字解析
      会夹杂大量非数字键，统一 fallback 到 sort_values 字典序原行为，不受影响。
    """
    try:
        col = df[x]
        # 兼容 object 与 pandas 2.x 的 string(StringDtype) 两种字符串列（真实传参为后者时会漏判）
        if len(col) and col.notna().any():
            sample = str(col.dropna().iloc[0])
            if re.match(r"^[\(\[]\s*[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+\s*[\)\]]$", sample):
                order = col.map(_interval_left).argsort(kind="stable")
                return df.iloc[order.values]
            # 分支②：纯数字 / 带 ≥ 前缀尾桶 —— 数值排序（区分普通文本列用解析成功率）
            non_na = col.dropna()
            if len(non_na):
                parsed = non_na.map(_numeric_key)
                # 仅当所有非空样本都能解析为有限数字时，才启用数值排序；
                # 否则视为普通文本列，保持 sort_values 原行为，避免误伤省份/类目等。
                if parsed.notna().all() and (parsed < float("inf")).all():
                    order = parsed.argsort(kind="stable")
                    return df.iloc[order.values]
        return df.sort_values(x)
    except Exception:
        return df


# 省份/地区/城市关键词 — 用于判断是否需要自动分组
_GEO_KEYWORDS = ['省', '市', '区', '县', '地区', '区域', '城市', '省份', '州', '国', '镇', '乡',
                  'province', 'city', 'region', 'area', 'district', 'state', 'country',
                  '部门', '科室', '单位', '组织', '机构', '类别', '类型', '分类', '分组']

_GEO_COL_KEYWORDS = ['省', '市', '区', '县', '地区', '区域', '城市', '省份', '地址', '位置',
                      'province', 'city', 'region', 'area', 'district', 'state', 'location',
                      '部门', '科室', '单位', '组织', '类别', '类型', '分类', '分组', '名称']


def _should_auto_group(df: pd.DataFrame, x: str) -> bool:
    """判断 X 轴列是否需要自动分组聚合
    
    条件：X 是分类列，且同一值出现多次（重复率 > 0），
    或者列名/值内容包含省份/地区等关键词
    """
    if x not in df.columns:
        return False
    
    col = df[x]
    # 1. 必须是分类列（object/category）
    dtype_str = str(col.dtype)
    if 'int' in dtype_str or 'float' in dtype_str or 'datetime' in dtype_str:
        return False
    
    # 2. 列名包含地区/分类关键词
    col_lower = x.lower()
    if any(kw in col_lower for kw in _GEO_COL_KEYWORDS):
        return True
    
    # 3. 值内容包含省份/地区关键词
    sample_vals = col.dropna().head(20).astype(str).tolist()
    if any(any(kw in v for kw in _GEO_KEYWORDS) for v in sample_vals):
        return True
    
    # 4. 同一值重复出现（非唯一映射）
    n_unique = col.nunique()
    n_total = len(col)
    if n_total > n_unique and n_total / n_unique > 1.2:
        return True
    
    return False


def _auto_groupby(df: pd.DataFrame, x: str, y: Optional[str] = None,
                   agg_func: str = 'sum') -> pd.DataFrame:
    """自动分组聚合：当 X 轴是省份/地区等分类列时，groupby X 并聚合 Y
    
    - 如果需要分组，返回 groupby 后的 DataFrame
    - 如果不需要分组，返回原始 DataFrame（不做任何修改）
    - agg_func 默认 'sum'，可选 'mean', 'count', 'max', 'min'
    """
    if not _should_auto_group(df, x):
        return df
    
    if y is None or y not in df.columns:
        # 没有 Y 列 → 做 value_counts
        counts = df[x].value_counts().reset_index()
        counts.columns = [x, 'count']
        return counts
    
    # groupby X，聚合 Y
    try:
        agg_df = df.groupby(x, as_index=False).agg({y: agg_func})
        return agg_df
    except Exception:
        # 聚合失败时返回原始数据
        return df


# ==================== 基础图表 ====================

def create_funnel_chart(df: pd.DataFrame, x: str = "", y: Optional[str] = None,
                        data: Optional[List[Dict[str, Any]]] = None,
                        title: str = "转化漏斗", **ignored) -> Dict[str, Any]:
    """创建漏斗图 ECharts option。

    支持两种数据来源（与 create_bar_chart 对齐）：
      - 列名方式：x=步骤名列，y=数值列（如 x="转化阶段", y="人数"）
      - 现成 data 方式：data=[{"name": "访问", "value": 1200}, ...]

    漏斗按 y/value 降序排列，更贴合"逐级流失"的直觉。
    """
    funnel_data: List[Dict[str, Any]] = []
    if data and isinstance(data, list) and len(data) > 0:
        for row in data:
            if isinstance(row, dict) and ("name" in row or "value" in row):
                funnel_data.append({
                    "name": str(row.get("name", row.get("label", ""))),
                    "value": float(row.get("value", row.get("count", 0)) or 0),
                })
    elif df is not None and not df.empty and x and y and x in df.columns and y in df.columns:
        tmp = df[[x, y]].copy()
        tmp[y] = pd.to_numeric(tmp[y], errors="coerce").fillna(0)
        for _, r in tmp.iterrows():
            funnel_data.append({"name": str(r[x]), "value": float(r[y])})

    if not funnel_data:
        return {}

    # 按数值降序，体现逐级流失
    funnel_data.sort(key=lambda d: d["value"], reverse=True)
    max_val = max((d["value"] for d in funnel_data), default=1) or 1

    option = {
        **_get_default_title(title),
        "tooltip": {**DARK_THEME["tooltip"], "trigger": "item", "formatter": "{b}: {c}"},
        "legend": {**DARK_THEME["legend"], "data": [d["name"] for d in funnel_data]},
        "toolbox": DARK_THEME["toolbox"],
        "series": [
            {
                "name": title,
                "type": "funnel",
                "left": "8%",
                "right": "8%",
                "top": 50,
                "bottom": 20,
                "min": 0,
                "max": max_val,
                "minSize": "12%",
                "maxSize": "100%",
                "sort": "descending",
                "gap": 3,
                "label": {"show": True, "position": "inside", "color": "#0F172A",
                          "fontSize": 12, "formatter": "{b}\n{c}"},
                "labelLine": {"show": False},
                "itemStyle": {"borderColor": "rgba(15,23,42,0.35)", "borderWidth": 1},
                "emphasis": {"label": {"fontSize": 14}},
                "data": funnel_data,
            }
        ],
    }
    return option


def create_bar_chart(df: pd.DataFrame, x: str, y: Optional[str] = None,
                     title: str = "柱状图", color: Optional[str] = None,
                     orientation: str = "v", **ignored) -> Dict[str, Any]:
    """创建柱状图 ECharts option — 自动对省份/地区列分组聚合"""
    if y is None:
        counts = df[x].value_counts().reset_index()
        counts.columns = [x, 'count']
        y = 'count'
        df_plot = counts
    else:
        # ★ 自动分组：X 轴是省份/地区等分类列时 groupby 聚合
        #   有 color 分组时跳过：分组意味着保留全部行作为不同系列，
        #   否则会按 x 聚合、丢失 color 列（导致 KeyError）。
        if color and color in df.columns:
            df_plot = df
        else:
            df_plot = _auto_groupby(df, x, y)

    df_plot = _sort_data(df_plot, x, y)
    x_data = [str(k) for k in pd.unique(df_plot[x])]
    y_values = df_plot[y].fillna(0).tolist()

    option = {
        **_get_default_title(title),
        "tooltip": DARK_THEME["tooltip"],
        "legend": DARK_THEME["legend"],
        "toolbox": DARK_THEME["toolbox"],
        "grid": {"top": 60, "left": 50, "right": 20, "bottom": 40},
    }

    if orientation == "h":
        option["yAxis"] = {"type": "category", "data": x_data, "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
        option["xAxis"] = {"type": "value", "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
    else:
        x_axis_name = _axis_name_or_none(x)
        y_axis_name = _axis_name_or_none(y)
        option["xAxis"] = {"type": "category", "data": x_data,
                           "axisLabel": {"rotate": 30 if len(x_data) > 8 else 0, "hideOverlap": True},
                           "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
        option["yAxis"] = {"type": "value",
                           "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
        # 仅当轴名非占位列名（__x__/__y__）时写入，避免占位符渲染成轴名乱飞
        if x_axis_name is not None:
            option["xAxis"]["name"] = x_axis_name
            option["xAxis"]["nameLocation"] = "middle"
            option["xAxis"]["nameGap"] = 30
        if y_axis_name is not None:
            option["yAxis"]["name"] = y_axis_name
            option["yAxis"]["nameLocation"] = "middle"
            option["yAxis"]["nameGap"] = 40

    if color and color in df.columns:
        groups = df_plot[color].unique()
        series_list = []
        for i, g in enumerate(groups):
            group_data = df_plot[df_plot[color] == g]
            g_y = group_data.set_index(x).reindex(df_plot[x].unique())[y].fillna(0).tolist()
            series_list.append({
                "name": str(g), "type": "bar",
                "data": g_y,
                "itemStyle": {"color": WARM_COLORS[i % len(WARM_COLORS)]}
            })
        option["series"] = series_list
    else:
        series_name = _axis_name_or_none(y)
        series_item = {"type": "bar", "data": [{"value": v, "itemStyle": {"color": WARM_COLORS[i % len(WARM_COLORS)]}}
                     for i, v in enumerate(y_values)]}
        # 仅当系列名非占位列名时写入，避免占位符污染 legend
        if series_name is not None:
            series_item["name"] = series_name
        option["series"] = [series_item]

    return option


def create_line_chart(df: pd.DataFrame, x: str, y: Optional[str] = None,
                      title: str = "折线图", color: Optional[str] = None, **ignored) -> Dict[str, Any]:
    """创建折线图 ECharts option — 自动对省份/地区列分组聚合"""
    if y is None:
        counts = df[x].value_counts().sort_index().reset_index()
        counts.columns = [x, 'count']
        y = 'count'
        df_plot = counts
    else:
        # ★ 自动分组：X 轴是省份/地区等分类列时 groupby 聚合
        #   有 color 分组时跳过：分组意味着保留全部行作为不同系列，
        #   否则会按 x 聚合、丢失 color 列（导致 KeyError）。
        if color and color in df.columns:
            df_plot = df
        else:
            df_plot = _auto_groupby(df, x, y)

    df_plot = _sort_data(df_plot, x, y)
    x_data = [str(k) for k in pd.unique(df_plot[x])]

    option = {
        **_get_default_title(title),
        "tooltip": DARK_THEME["tooltip"],
        "legend": DARK_THEME["legend"],
        "toolbox": DARK_THEME["toolbox"],
        "grid": {"top": 60, "left": 50, "right": 20, "bottom": 40},
        "xAxis": {"type": "category", "data": x_data, "boundaryGap": False,
                  "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
        "yAxis": {"type": "value", "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
    }

    if color and color in df.columns:
        groups = df_plot[color].unique()
        series_list = []
        for i, g in enumerate(groups):
            group_data = df_plot[df_plot[color] == g]
            g_y = group_data.set_index(x).reindex(df_plot[x].unique())[y].fillna(0).tolist()
            series_list.append({
                "name": str(g), "type": "line",
                "data": g_y, "smooth": True,
                "lineStyle": {"color": WARM_COLORS[i % len(WARM_COLORS)]},
                "itemStyle": {"color": WARM_COLORS[i % len(WARM_COLORS)]},
            })
        option["series"] = series_list
    else:
        series_name = _axis_name_or_none(y)
        series_item = {
            "type": "line",
            "data": df_plot[y].fillna(0).tolist(),
            "smooth": True,
            "lineStyle": {"color": WARM_COLORS[0]},
            "itemStyle": {"color": WARM_COLORS[0]},
        }
        if series_name is not None:
            series_item["name"] = series_name
        option["series"] = [series_item]
    return option


def create_area_chart(df: pd.DataFrame, x: str, y: Optional[str] = None,
                      title: str = "面积图", color: Optional[str] = None, **ignored) -> Dict[str, Any]:
    """创建面积图"""
    option = create_line_chart(df, x, y, title, color)
    for s in option.get("series", []):
        s["areaStyle"] = {"opacity": 0.3}
    return option


def create_scatter_chart(df: pd.DataFrame, x: str, y: Optional[str] = None,
                         title: str = "散点图", color: Optional[str] = None, **ignored) -> Dict[str, Any]:
    """创建散点图"""
    if y is None:
        raise ValueError("散点图需要同时指定 X 轴和 Y 轴")

    option = {
        **_get_default_title(title),
        "tooltip": DARK_THEME["tooltip"],
        "legend": DARK_THEME["legend"],
        "toolbox": DARK_THEME["toolbox"],
        "grid": {"top": 60, "left": 50, "right": 20, "bottom": 40},
        "xAxis": {"type": "value", "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
        "yAxis": {"type": "value", "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
    }

    if color and color in df.columns:
        groups = df[color].unique()
        series_list = []
        for i, g in enumerate(groups):
            group_data = df[df[color] == g]
            pts = [
                [float(row[x]), float(row[y])]
                for _, row in group_data.iterrows()
                if pd.notna(row[x]) and pd.notna(row[y])
            ]
            series_list.append({
                "name": str(g), "type": "scatter",
                "data": pts,
                "symbolSize": 8,
                "itemStyle": {"color": WARM_COLORS[i % len(WARM_COLORS)]}
            })
        option["series"] = series_list
    else:
        pts = [
            [float(row[x]), float(row[y])]
            for _, row in df.iterrows()
            if pd.notna(row[x]) and pd.notna(row[y])
        ]
        option["series"] = [{
            "name": f"{x}-{y}", "type": "scatter",
            "data": pts,
            "symbolSize": 8,
            "itemStyle": {"color": WARM_COLORS[0]}
        }]
    return option


def create_pie_chart(df: pd.DataFrame, names: Optional[str] = None, values: Optional[str] = None,
                     x: Optional[str] = None, y: Optional[str] = None,
                     title: str = "饼图", **ignored) -> Dict[str, Any]:
    """创建饼图"""
    names = names or x
    values = values or y

    if names is None:
        raise ValueError("饼图需要指定分类列")

    if values:
        data = df.groupby(names)[values].sum().reset_index()
    else:
        data = df[names].value_counts().reset_index()
        data.columns = [names, 'count']
        values = 'count'

    pie_data = []
    for i, (_, row) in enumerate(data.iterrows()):
        pie_data.append({
            "name": str(row[names]),
            "value": float(row[values]),
            "itemStyle": {"color": WARM_COLORS[i % len(WARM_COLORS)]}
        })

    return {
        **_get_default_title(title),
        "tooltip": DARK_THEME["tooltip"],
        "legend": DARK_THEME["legend"],
        "toolbox": DARK_THEME["toolbox"],
        "series": [{
            "type": "pie",
            "radius": ["35%", "65%"],
            "center": ["50%", "50%"],
            "emphasis": {
                "label": {"fontSize": 18, "fontWeight": "bold"},
                "scaleSize": 10
            },
            "data": pie_data,
            "label": {"color": "#94a3b8"},
            "labelLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}},
        }]
    }


def create_histogram(df: pd.DataFrame, x: str, title: str = "直方图", y: Optional[str] = None, **ignored) -> Dict[str, Any]:
    """创建直方图（用 bar 模拟，手动分箱 — ECharts 无内置 histogram 类型）"""
    values = df[x].dropna()
    if len(values) == 0:
        raise ValueError("直方图需要数值数据")

    # 自动计算分箱数
    n_bins = min(20, max(5, int(len(values) ** 0.5)))
    min_val, max_val = float(values.min()), float(values.max())
    bin_width = (max_val - min_val) / n_bins if max_val > min_val else 1

    bins = [min_val + i * bin_width for i in range(n_bins + 1)]
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(n_bins)]

    # 统计每个区间的频次
    counts = [0] * n_bins
    for v in values:
        idx = min(int((float(v) - min_val) / bin_width), n_bins - 1)
        counts[idx] += 1

    return {
        **_get_default_title(title),
        "tooltip": DARK_THEME["tooltip"],
        "toolbox": DARK_THEME["toolbox"],
        "grid": {"top": 60, "left": 50, "right": 20, "bottom": 50},
        "xAxis": {
            "type": "category", "data": labels, "name": x,
            "axisLabel": {"rotate": 30, "fontSize": 10},
            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}
        },
        "yAxis": {
            "type": "value", "name": y if y else "频次",
            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}
        },
        "series": [{
            "name": "频次", "type": "bar",
            "data": counts,
            "itemStyle": {"color": WARM_COLORS[0]},
            "barWidth": "90%",
        }]
    }


def create_box_plot(df: pd.DataFrame, y: str, x: Optional[str] = None,
                    title: str = "箱线图", **ignored) -> Dict[str, Any]:
    """创建箱线图"""
    option = {
        **_get_default_title(title),
        "tooltip": DARK_THEME["tooltip"],
        "toolbox": DARK_THEME["toolbox"],
        "grid": {"top": 60, "left": 50, "right": 20, "bottom": 40},
    }

    if x and x in df.columns:
        groups = df[x].dropna().unique()
        x_data = [str(g) for g in groups]
        box_data = []
        for i, g in enumerate(groups):
            group_vals = df[df[x] == g][y].dropna().tolist()
            box_data.append({
                "name": str(g),
                "value": group_vals,
            })
        option["xAxis"] = {"type": "category", "data": x_data, "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
        option["yAxis"] = {"type": "value", "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
        option["series"] = [{
            "name": y, "type": "boxplot",
            "data": [
                {
                    "name": str(groups[i]),
                    "value": [
                        np.percentile(v, 0) if v else 0,    # min
                        np.percentile(v, 25) if v else 0,   # Q1
                        np.percentile(v, 50) if v else 0,   # median
                        np.percentile(v, 75) if v else 0,   # Q3
                        np.percentile(v, 100) if v else 0,  # max
                    ]
                }
                for i, v in enumerate([df[df[x] == g][y].dropna().tolist() for g in groups])
            ],
            "itemStyle": {"color": WARM_COLORS[0], "borderColor": WARM_COLORS[1]},
            "boxWidth": [20, 40],
        }]
    else:
        vals = df[y].dropna().tolist()
        option["xAxis"] = {"type": "category", "data": [y], "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
        option["yAxis"] = {"type": "value", "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
        option["series"] = [{
            "name": y, "type": "boxplot",
            "data": [[
                np.percentile(vals, 0), np.percentile(vals, 25),
                np.percentile(vals, 50), np.percentile(vals, 75),
                np.percentile(vals, 100)
            ]],
            "itemStyle": {"color": WARM_COLORS[0], "borderColor": WARM_COLORS[1]},
        }]
    return option


def create_heatmap(df: pd.DataFrame, title: str = "相关性热力图", **ignored) -> Optional[Dict[str, Any]]:
    """创建相关性热力图"""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return None

    corr = numeric_df.corr()
    x_data = corr.columns.tolist()
    y_data = corr.index.tolist()
    data = []
    max_val = 0
    for i, row_name in enumerate(y_data):
        for j, col_name in enumerate(x_data):
            v = round(float(corr.iloc[i, j]), 2)
            data.append([j, i, v])
            max_val = max(max_val, abs(v))

    return {
        **_get_default_title(title),
        "_heatmap_kind": "correlation",  # 相关性热力图：超限须正确重映射坐标
        "tooltip": DARK_THEME["tooltip"],
        "toolbox": DARK_THEME["toolbox"],
        "grid": {"top": 60, "left": 80, "right": 40, "bottom": 40},
        "xAxis": {"type": "category", "data": x_data, "axisLabel": {"rotate": 30},
                  "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}, "position": "top"},
        "yAxis": {"type": "category", "data": y_data,
                  "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
        "visualMap": {
            "min": -max_val, "max": max_val,
            "inRange": {"color": ["#23304E", "#1E3A8A", "#0369a1", "#38BDF8", "#7DD3FC"]},
            "text": ["正相关", "负相关"],
            "textStyle": {"color": "#94a3b8"},
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 0,
        },
        "series": [{
            "type": "heatmap", "data": data,
            "label": {"show": True, "color": "#e2e8f0", "fontSize": 11,
                      "formatter": "{@[2]}"},
            "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.5)"}},
            "itemStyle": {"borderColor": "#1e1e3a", "borderWidth": 1},
        }]
    }


def create_radar_chart(df: pd.DataFrame, x: Optional[str] = None, y: Optional[str] = None,
                       title: str = "雷达图", **ignored) -> Dict[str, Any]:
    """创建雷达图"""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    dim_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not dim_cols:
        raise ValueError("雷达图需要数值列")

    if len(dim_cols) > 8:
        dim_cols = dim_cols[:8]  # 雷达图最多8个维度

    if title == "雷达图":
        title = f"雷达图（{' · '.join(dim_cols)}）"

    group_col = x if x and x in cat_cols else (cat_cols[0] if cat_cols else None)

    indicator = [{"name": c, "max": float(df[c].max() * 1.2)} for c in dim_cols]

    if group_col:
        agg = df.groupby(group_col)[dim_cols].mean().reset_index()
        series_list = []
        for i, (_, row) in enumerate(agg.iterrows()):
            series_list.append({
                "name": str(row[group_col]),
                "type": "radar",
                "data": [{"value": [float(row[c]) for c in dim_cols],
                          "name": str(row[group_col])}],
                "lineStyle": {"color": WARM_COLORS[i % len(WARM_COLORS)]},
                "areaStyle": {"color": WARM_COLORS[i % len(WARM_COLORS)], "opacity": 0.15},
                "symbolSize": 6,
            })
    else:
        means = [float(df[c].mean()) for c in dim_cols]
        series_list = [{
            "name": "平均值", "type": "radar",
            "data": [{"value": means, "name": "平均值"}],
            "lineStyle": {"color": WARM_COLORS[0]},
            "areaStyle": {"color": WARM_COLORS[0], "opacity": 0.15},
            "symbolSize": 6,
        }]

    return {
        **_get_default_title(title),
        "tooltip": DARK_THEME["tooltip"],
        "legend": DARK_THEME["legend"],
        "toolbox": DARK_THEME["toolbox"],
        "radar": {
            "indicator": indicator,
            "center": ["50%", "55%"],
            "radius": "65%",
            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}},
            "splitArea": {"areaStyle": {"color": ["rgba(56,189,248,0.03)", "rgba(56,189,248,0.06)"]}},
        },
        "series": series_list,
    }


def create_stacked_bar(df: pd.DataFrame, x: str, y: Optional[str] = None,
                       color: Optional[str] = None, title: str = "堆叠柱状图", **ignored) -> Dict[str, Any]:
    """创建堆叠柱状图"""
    option = create_bar_chart(df, x, y, title, color)
    for s in option.get("series", []):
        s["stack"] = "total"
    return option


def create_waterfall(df: pd.DataFrame, x: str, y: Optional[str] = None,
                     title: str = "瀑布图", **ignored) -> Dict[str, Any]:
    """创建瀑布图（用堆叠柱状图模拟）— 自动对省份/地区列分组聚合"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if y is None:
        if not numeric_cols:
            raise ValueError("瀑布图需要数值列")
        y = numeric_cols[0]

    # ★ 自动分组：X 轴是省份/地区等分类列时 groupby 聚合
    df_plot = _auto_groupby(df, x, y)

    values = df_plot[y].fillna(0).tolist()
    x_data = df_plot[x].astype(str).tolist()

    # 瀑布图：base = 前几项累积和
    base = [0]
    for i in range(len(values) - 1):
        base.append(base[-1] + values[i])

    option = {
        **_get_default_title(title),
        "tooltip": DARK_THEME["tooltip"],
        "toolbox": DARK_THEME["toolbox"],
        "grid": {"top": 60, "left": 60, "right": 20, "bottom": 40},
        "xAxis": {"type": "category", "data": x_data,
                  "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
        "yAxis": {"type": "value", "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
        "series": [
            {"name": "占位", "type": "bar", "stack": "waterfall",
             "data": base, "itemStyle": {"color": "transparent"},
             "label": {"show": False}},
            {"name": "变化", "type": "bar", "stack": "waterfall",
             "data": [{"value": v, "itemStyle": {"color": GALAXY["primary"] if v >= 0 else GALAXY["danger"]}}
                      for v in values]}
        ]
    }
    return option


def create_bubble_chart(df: pd.DataFrame, x: str, y: Optional[str] = None,
                        size: Optional[str] = None, color: Optional[str] = None,
                        title: str = "气泡图", **ignored) -> Dict[str, Any]:
    """创建气泡图"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not x or y is None:
        raise ValueError("气泡图需要 X 轴和 Y 轴")
    if size is None and len(numeric_cols) >= 3:
        size = numeric_cols[2]

    size_vals = df[size].fillna(10).tolist() if size else [10] * len(df)
    max_s = max(size_vals) if size_vals else 1
    scaled_sizes = [max(5, min(50, (s / max_s) * 40)) for s in size_vals]

    option = {
        **_get_default_title(title),
        "tooltip": DARK_THEME["tooltip"],
        "toolbox": DARK_THEME["toolbox"],
        "grid": {"top": 60, "left": 50, "right": 20, "bottom": 40},
        "xAxis": {"type": "value", "name": x, "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
        "yAxis": {"type": "value", "name": y, "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
    }

    if color and color in df.columns:
        groups = df[color].unique()
        series_list = []
        for i, g in enumerate(groups):
            group_data = df[df[color] == g]
            pts = []
            for idx, (_, row) in enumerate(group_data.iterrows()):
                if pd.notna(row[x]) and pd.notna(row[y]):
                    pts.append([float(row[x]), float(row[y]), scaled_sizes[idx % len(scaled_sizes)] if size else 10])
            series_list.append({
                "name": str(g), "type": "scatter",
                "data": pts,
                "symbolSize": "function(data) { return data[2]; }",
                "itemStyle": {"color": WARM_COLORS[i % len(WARM_COLORS)], "opacity": 0.7},
            })
        option["series"] = series_list
    else:
        pts = []
        for idx, (_, row) in enumerate(df.iterrows()):
            if pd.notna(row[x]) and pd.notna(row[y]):
                pts.append([float(row[x]), float(row[y]), scaled_sizes[idx] if size else 10])
        option["series"] = [{
            "name": f"{x}-{y}", "type": "scatter",
            "data": pts,
            "symbolSize": "function(data) { return data[2]; }",
            "itemStyle": {"color": WARM_COLORS[0], "opacity": 0.7},
        }]
    return option


def create_bubble_matrix(df: pd.DataFrame, x: str, y: Optional[str] = None,
                         title: str = "气泡矩阵图", color: Optional[str] = None,
                         size_col: str = "人数", **ignored) -> Dict[str, Any]:
    """气泡矩阵图（流失预警专用）：类目轴 x=价值层、y=流失状态，气泡大小∝人数，颜色区分系列。

    约定 data 列：x 列（价值层）、y 列（流失状态）、size_col（人数）、color 列（系列，如挽回优先级）。
    """
    if x is None or y is None:
        raise ValueError("气泡矩阵图需要 X 轴(价值层) 与 Y 轴(流失状态) 两个类目列")
    if size_col not in df.columns:
        raise ValueError(f"气泡矩阵图需要人数列[{size_col}]")
    x_cats = [str(v) for v in pd.unique(df[x])]
    y_cats = [str(v) for v in pd.unique(df[y])]

    if color and color in df.columns:
        groups = [str(v) for v in pd.unique(df[color])]
    else:
        groups = ["__all__"]

    _PRIO_COLOR = {
        "紧急挽回": GALAXY["danger"],       # 红 #FB7185
        "重点防护": "#FB923C",              # 橙
        "标准召回": GALAXY["warning"],      # 金 #FBBF24
        "常规跟进": GALAXY["primary"],      # 蓝 #38BDF8
        "停止触达": GALAXY["text_secondary"],  # 蓝灰 #94A3B8
        "常规维持": "#64748B",              # 更深中性灰
    }
    # 确定性取色：非预设分组按「分组名排序后的索引」取色板。
    # 禁用 hash()：Python 字符串 hash 每次进程启动随机化（PYTHONHASHSEED），
    # 同一份数据两次运行颜色会变。
    _other_groups = sorted(g for g in groups if g not in _PRIO_COLOR)
    _group_color = {g: BLUE_PALETTE[i % len(BLUE_PALETTE)]
                    for i, g in enumerate(_other_groups)}

    def _color_for(g: str) -> str:
        if g in _PRIO_COLOR:
            return _PRIO_COLOR[g]
        return _group_color.get(g, BLUE_PALETTE[0])

    size_vals = df[size_col].fillna(0).astype(float)
    max_s = max(float(size_vals.max()), 1e-9)
    data_pts = []
    for _, row in df.iterrows():
        sx = str(row[x]); sy = str(row[y])
        sz = float(row[size_col]) if pd.notna(row[size_col]) else 0.0
        sym = max(8, min(60, (sz / max_s) * 50))
        g = str(row[color]) if (color and color in df.columns) else "__all__"
        data_pts.append({
            "value": [sx, sy, sz],
            "symbolSize": sym,
            "itemStyle": {"color": _color_for(g), "opacity": 0.8},
        })

    option = {
        **_get_default_title(title),
        "tooltip": DARK_THEME["tooltip"],
        "toolbox": DARK_THEME["toolbox"],
        "grid": {"top": 60, "left": 90, "right": 30, "bottom": 50},
        "xAxis": {"type": "category", "data": x_cats, "name": x,
                  "nameLocation": "middle", "nameGap": 30,
                  "axisLine": {"lineStyle": {"color": GALAXY["grid"]}}},
        "yAxis": {"type": "category", "data": y_cats, "name": y,
                  "nameLocation": "middle", "nameGap": 55,
                  "axisLine": {"lineStyle": {"color": GALAXY["grid"]}}},
        "series": [{
            "type": "scatter",
            "data": data_pts,
            "label": {"show": True, "formatter": "{@[2]}",
                      "color": GALAXY["text_secondary"], "fontSize": 10},
        }],
    }
    return option


def create_horizontal_bar(df: pd.DataFrame, x: str, y: Optional[str] = None,
                          title: str = "横向柱状图", **ignored) -> Dict[str, Any]:
    """横向柱状图（流失归因专用）：类目轴=维度取值，数值轴=偏移值；按正负红绿着色。

    约定 data 列：x 列（维度取值，类目）、y 列（偏移值，数值，可正可负，单位 pp）。
    """
    if x is None:
        return {}
    if y is None:
        if x in df.columns:
            dft = df[x].value_counts().reset_index()
            dft.columns = [x, "count"]
            xcats = [str(v) for v in dft[x]]
            ydata = dft["count"].tolist()
        else:
            return {}
    else:
        xcats = [str(v) for v in df[x].tolist()]
        ydata = [float(v) if pd.notna(v) else 0.0 for v in df[y].tolist()]

    bar_data = []
    for val in ydata:
        if val > 0:
            col = GALAXY["danger"]
        elif val < 0:
            col = GALAXY["success"]
        else:
            col = GALAXY["text_secondary"]
        bar_data.append({"value": round(val, 1), "itemStyle": {"color": col}})

    option = {
        **_get_default_title(title),
        "tooltip": {**DARK_THEME["tooltip"], "trigger": "axis",
                    "axisPointer": {"type": "shadow"}},
        "toolbox": DARK_THEME["toolbox"],
        "grid": {"top": 60, "left": 130, "right": 50, "bottom": 40},
        "xAxis": {"type": "value", "name": "偏移 (pp)", "nameLocation": "middle",
                  "nameGap": 30, "axisLine": {"lineStyle": {"color": GALAXY["grid"]}}},
        "yAxis": {"type": "category", "data": xcats, "inverse": True,
                  "axisLine": {"lineStyle": {"color": GALAXY["grid"]}}},
        "series": [{
            "type": "bar", "data": bar_data,
            "label": {"show": True, "position": "right",
                      "color": GALAXY["text_secondary"], "fontSize": 10},
        }],
    }
    return option


def create_hbar_family(df: pd.DataFrame, x: str, y: Optional[str] = None,
                       title: str = "维度偏移图族", color: Optional[str] = None,
                       **ignored) -> Dict[str, Any]:
    """横向柱状图族（流失归因 F 专用）：按维度列分组，每维度一张横向偏移图。

    约定 data 列：color 列（默认「维度」，分组键）、x 列（维度取值）、y 列（偏移值 pp）。
    返回 {维度名: 子图option} 字典（非标准 ECharts option）：
      - 前端 chart_type=="hbar_family" 时遍历该 map 渲染 N 个图表实例；
      - create_chart 的 _cap_option_data 对无 series 键的 dict 原样放行，
        故在此对每个子 option 单独做截断护栏。
    """
    group_col = color if color and color in df.columns else "维度"
    if group_col not in df.columns or x is None:
        return {}
    family: Dict[str, Any] = {}
    for dim, sub in df.groupby(group_col, sort=False):
        sub_option = create_horizontal_bar(
            sub.reset_index(drop=True), x=x, y=y,
            title=f"{dim}：流失维度偏移（流失群−正常群，pp）",
        )
        if sub_option:
            family[str(dim)] = _cap_option_data(sub_option)
    return family


def create_treemap(df: pd.DataFrame, x: Optional[str] = None, y: Optional[str] = None,
                   title: str = "树状图", **ignored) -> Dict[str, Any]:
    """创建树状图"""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    path_col = x if x and x in cat_cols else (cat_cols[0] if cat_cols else None)
    values_col = y if y and y in numeric_cols else (numeric_cols[0] if numeric_cols else None)

    if not path_col:
        raise ValueError("树状图需要分类列")

    if values_col:
        agg = df.groupby(path_col)[values_col].sum().reset_index()
    else:
        agg = df[path_col].value_counts().reset_index()
        agg.columns = [path_col, 'count']
        values_col = 'count'

    treemap_data = []
    for i, (_, row) in enumerate(agg.iterrows()):
        treemap_data.append({
            "name": str(row[path_col]),
            "value": float(row[values_col]),
            "itemStyle": {"color": WARM_COLORS[i % len(WARM_COLORS)]}
        })

    return {
        **_get_default_title(title),
        "tooltip": DARK_THEME["tooltip"],
        "series": [{
            "type": "treemap",
            "data": treemap_data,
            "label": {"color": "#e2e8f0"},
            "upperLabel": {"show": True, "height": 30},
            "itemStyle": {"borderColor": "#1e1e3a", "borderWidth": 2},
        }]
    }


# ==================== 3D 地图（ECharts GL） ====================

# GeoJSON 省份全称映射（DataV 格式）
_GEO_PROVINCE_NAMES = {
    '北京': '北京市', '北京市': '北京市',
    '上海': '上海市', '上海市': '上海市',
    '天津': '天津市', '天津市': '天津市',
    '重庆': '重庆市', '重庆市': '重庆市',
    '河北': '河北省', '河北省': '河北省',
    '山西': '山西省', '山西省': '山西省',
    '辽宁': '辽宁省', '辽宁省': '辽宁省',
    '吉林': '吉林省', '吉林省': '吉林省',
    '黑龙江': '黑龙江省', '黑龙江省': '黑龙江省',
    '江苏': '江苏省', '江苏省': '江苏省',
    '浙江': '浙江省', '浙江省': '浙江省',
    '安徽': '安徽省', '安徽省': '安徽省',
    '福建': '福建省', '福建省': '福建省',
    '江西': '江西省', '江西省': '江西省',
    '山东': '山东省', '山东省': '山东省',
    '河南': '河南省', '河南省': '河南省',
    '湖北': '湖北省', '湖北省': '湖北省',
    '湖南': '湖南省', '湖南省': '湖南省',
    '广东': '广东省', '广东省': '广东省',
    '海南': '海南省', '海南省': '海南省',
    '四川': '四川省', '四川省': '四川省',
    '贵州': '贵州省', '贵州省': '贵州省',
    '云南': '云南省', '云南省': '云南省',
    '陕西': '陕西省', '陕西省': '陕西省',
    '甘肃': '甘肃省', '甘肃省': '甘肃省',
    '青海': '青海省', '青海省': '青海省',
    '台湾': '台湾省', '台湾省': '台湾省',
    '广西': '广西壮族自治区', '广西壮族自治区': '广西壮族自治区',
    '内蒙古': '内蒙古自治区', '内蒙古自治区': '内蒙古自治区',
    '西藏': '西藏自治区', '西藏自治区': '西藏自治区',
    '宁夏': '宁夏回族自治区', '宁夏回族自治区': '宁夏回族自治区',
    '新疆': '新疆维吾尔自治区', '新疆维吾尔自治区': '新疆维吾尔自治区',
    '香港': '香港特别行政区', '香港特别行政区': '香港特别行政区',
    '澳门': '澳门特别行政区', '澳门特别行政区': '澳门特别行政区',
}

# 省份中心坐标（用于 bar3D 放置柱子）
_PROVINCE_CENTROIDS = {
    '北京市': [116.4, 39.9], '天津市': [117.2, 39.1], '上海市': [121.5, 31.2], '重庆市': [106.5, 29.6],
    '河北省': [114.5, 38.0], '山西省': [112.5, 37.9], '辽宁省': [123.4, 41.8], '吉林省': [125.3, 43.9],
    '黑龙江省': [126.6, 45.8], '江苏省': [119.8, 33.0], '浙江省': [120.2, 30.3], '安徽省': [117.3, 31.8],
    '福建省': [119.3, 26.1], '江西省': [115.9, 27.7], '山东省': [117.0, 36.7], '河南省': [113.7, 33.9],
    '湖北省': [112.4, 31.2], '湖南省': [112.0, 27.1], '广东省': [113.5, 23.5], '海南省': [110.0, 19.2],
    '四川省': [102.2, 30.6], '贵州省': [106.7, 26.6], '云南省': [102.7, 25.0], '陕西省': [108.9, 34.3],
    '甘肃省': [103.8, 36.1], '青海省': [96.0, 36.5], '台湾省': [121.0, 24.0],
    '广西壮族自治区': [108.3, 22.8], '内蒙古自治区': [111.8, 40.8], '西藏自治区': [89.1, 31.5],
    '宁夏回族自治区': [106.3, 37.1], '新疆维吾尔自治区': [85.6, 42.1],
    '香港特别行政区': [114.2, 22.3],     '澳门特别行政区': [113.5, 22.2],
}

# ★ 省份→大区映射（用于按地区着色地图）
_PROVINCE_TO_REGION = {
    # 华东
    '上海市': '华东', '江苏省': '华东', '浙江省': '华东', '安徽省': '华东',
    '福建省': '华东', '江西省': '华东', '山东省': '华东',
    # 华北
    '北京市': '华北', '天津市': '华北', '河北省': '华北', '山西省': '华北',
    '内蒙古自治区': '华北',
    # 华中
    '河南省': '华中', '湖北省': '华中', '湖南省': '华中',
    # 华南
    '广东省': '华南', '广西壮族自治区': '华南', '海南省': '华南',
    # 西南
    '重庆市': '西南', '四川省': '西南', '贵州省': '西南', '云南省': '西南',
    '西藏自治区': '西南',
    # 东北
    '辽宁省': '东北', '吉林省': '东北', '黑龙江省': '东北',
    # 西北
    '陕西省': '西北', '甘肃省': '西北', '青海省': '西北',
    '宁夏回族自治区': '西北', '新疆维吾尔自治区': '西北',
    # 港澳台
    '香港特别行政区': '港澳台', '澳门特别行政区': '港澳台', '台湾省': '港澳台',
}

# ★ 大区中心坐标（用于地区模式下散点/标签）
_REGION_CENTROIDS = {
    '华东': [118.5, 32.5],
    '华北': [115.0, 40.0],
    '华中': [113.0, 32.0],
    '华南': [112.0, 23.0],
    '西南': [104.0, 29.0],
    '东北': [125.0, 44.0],
    '西北': [97.0, 38.0],
    '港澳台': [118.0, 24.0],
}


_CITY_TO_PROVINCE = {
    '深圳': '广东省', '广州': '广东省', '东莞': '广东省', '佛山': '广东省',
    '杭州': '浙江省', '宁波': '浙江省', '温州': '浙江省',
    '南京': '江苏省', '苏州': '江苏省', '无锡': '江苏省',
    '成都': '四川省', '武汉': '湖北省', '西安': '陕西省',
    '郑州': '河南省', '青岛': '山东省', '济南': '山东省',
    '长沙': '湖南省', '合肥': '安徽省', '福州': '福建省',
    '厦门': '福建省', '南昌': '江西省', '大连': '辽宁省',
    '沈阳': '辽宁省', '长春': '吉林省', '哈尔滨': '黑龙江省',
    '石家庄': '河北省', '太原': '山西省', '南宁': '广西壮族自治区',
    '昆明': '云南省', '贵阳': '贵州省', '兰州': '甘肃省',
    '呼和浩特': '内蒙古自治区', '乌鲁木齐': '新疆维吾尔自治区',
    '拉萨': '西藏自治区', '银川': '宁夏回族自治区',
    '海口': '海南省', '台北': '台湾省', '高雄': '台湾省',
}

def _to_geo_name(name: str) -> str:
    """将各种形式的省份名/城市名转为 GeoJSON 标准名称"""
    name_str = str(name).strip()
    
    if name_str in _CITY_TO_PROVINCE:
        return _CITY_TO_PROVINCE[name_str]
    
    cleaned = name_str.replace('省', '').replace('市', '').replace('自治区', '').replace('特别行政区', '').replace('壮族', '').replace('回族', '').replace('维吾尔', '').strip()
    return _GEO_PROVINCE_NAMES.get(name_str, _GEO_PROVINCE_NAMES.get(cleaned, name_str))


def _province_short_name(params: dict) -> str:
    """geo3D label formatter：将省份全称缩短为 2-3 字简称"""
    name = str(params.get('name', '') or params.get('properties', {}).get('name', ''))
    # 移除后缀
    short = name.replace('省', '').replace('市', '').replace('自治区', '').replace('特别行政区', '')
    short = short.replace('壮族', '').replace('回族', '').replace('维吾尔', '')
    short = short.strip()
    # 缩短长名
    if short == '内蒙古':
        short = '蒙'
    elif short == '黑龙江':
        short = '黑'
    elif len(short) > 3:
        short = short[:2]
    return short


def _build_geo3d_regions(map_data: list, min_val: float, max_val: float) -> list:
    """为 geo3D 构建 regions 配置，按数据值给省份上色"""
    val_range = max_val - min_val or 1
    gradient_colors = ["#13243F", "#0c4a6e", "#0369a1", "#0ea5e9", "#38BDF8", "#67E8F9", "#22D3EE"]
    regions = []
    for item in map_data:
        name = item["name"]
        val = item["value"]
        # 计算颜色索引
        ratio = (val - min_val) / val_range
        idx = int(ratio * (len(gradient_colors) - 1))
        idx = max(0, min(idx, len(gradient_colors) - 1))
        color = gradient_colors[idx]
        regions.append({
            "name": name,
            "itemStyle": {"areaColor": color, "opacity": 0.8},
            "label": {"show": True, "color": "#e2e8f0", "fontSize": 11, "formatter": "{b}"},
        })
    return regions


def _format_number(n: float) -> str:
    """格式化数字：万/亿 单位"""
    if abs(n) >= 1e8:
        return f"{n/1e8:.2f}亿"
    elif abs(n) >= 1e4:
        return f"{n/1e4:.1f}万"
    elif abs(n) >= 1000:
        return f"{n:,.0f}"
    elif abs(n) >= 1:
        return f"{n:.2f}"
    else:
        return f"{n:.4f}"


def _color_by_value(val: float, min_val: float, max_val: float) -> str:
    """星空渐变色：深空紫 → 星云蓝 → 星光青 → 超新星白"""
    gradient = [
        "#020617",  # 深空
        "#0B1B3A",  # 暗星云蓝
        "#0c4a6e",  # 蓝星云
        "#0369a1",  # 蓝星
        "#0ea5e9",  # 亮蓝
        "#38BDF8",  # 星光蓝
        "#22D3EE",  # 极光青
        "#67E8F9",  # 亮星蓝
        "#7DD3FC",  # 星白蓝
        "#BFE9FF",  # 浅蓝
        "#E6F7FF",  # 星白
    ]
    ratio = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    idx = int(ratio * (len(gradient) - 1))
    idx = max(0, min(idx, len(gradient) - 1))
    return gradient[idx]


def create_gl_map(df: pd.DataFrame, x: Optional[str] = None, y: Optional[str] = None,
                  title: str = "数据地图", **ignored) -> Dict[str, Any]:
    """星空主题 2D 数据地图：geo + effectScatter，省份标签 + 星光散点
    
    自动回退：如果 X 列值无法匹配 GeoJSON 省份名（如 X='地区' 值是华东/华北），
    会自动查找并切换到「省份」/「城市」等能匹配的列。
    """

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not num_cols:
        raise ValueError("地图需要数值列")

    region_col = x if x and x in df.columns else (cat_cols[0] if cat_cols else None)
    value_col = y if y and y in num_cols else num_cols[0]
    if not region_col:
        raise ValueError("未找到地区/省份列，请选择包含省份或城市名称的列作为 X 轴")

    # ★ 检查当前 X 列的值是否能匹配 GeoJSON 省份名
    # 如果匹配率很低（如 X='地区' 时值是华东/华北），自动回退到「省份」列
    sample_names = df[region_col].dropna().astype(str).unique()[:20]
    match_count = sum(1 for n in sample_names if _to_geo_name(n) in _PROVINCE_CENTROIDS)
    match_rate = match_count / len(sample_names) if len(sample_names) > 0 else 0
    
    # ★ 地区→省份展开：X 值不是省份名时（如华东/华北），按大区聚合后展开到各省份
    is_region_mode = False
    region_province_map = {}   # region_name → [province_geo_names]
    province_region_map = {}   # province_geo_name → region_name (for tooltip)
    _original_region_col = region_col

    if match_rate < 0.3:
        # 尝试找 df 中的省份列来展开地区
        province_src_col = None
        for c in df.columns:
            cl = c.lower()
            if any(kw in cl for kw in ['省份', 'province', '省市', '城市', 'city']):
                province_src_col = c
                break

        if province_src_col and province_src_col != region_col:
            # 从数据中建立 地区→省份 映射
            _data_map = {}
            for _, row in df[[region_col, province_src_col]].drop_duplicates().iterrows():
                rname = str(row[region_col])
                pname = str(row[province_src_col])
                pgeo = _to_geo_name(pname)
                if pgeo and pgeo in _PROVINCE_CENTROIDS:
                    _data_map.setdefault(rname, []).append(pgeo)

            if _data_map:
                # 按地区聚合值
                region_agg = df.groupby(region_col, as_index=False)[value_col].sum()
                region_vals = dict(zip(region_agg[region_col].astype(str), region_agg[value_col]))

                # 展开：每个省份继承其所属地区的值
                expanded_rows = []
                for rname, prov_list in _data_map.items():
                    rval = region_vals.get(rname, 0)
                    for pgeo in prov_list:
                        expanded_rows.append({'province_geo': pgeo, value_col: rval})
                        province_region_map[pgeo] = rname
                    region_province_map[rname] = prov_list

                df_agg = pd.DataFrame(expanded_rows)
                region_col = 'province_geo'
                is_region_mode = True
                import logging
                logging.getLogger(__name__).info(
                    f"地图地区展开：X「{_original_region_col}」→ 省份级着色，"
                    f"{len(_data_map)} 个大区映射到 {len(expanded_rows)} 个省份"
                )

    # 非 region mode 的常规聚合
    if not is_region_mode:
        try:
            df_agg = df.groupby(region_col, as_index=False).agg({value_col: 'sum'})
        except Exception:
            df_agg = df[[region_col, value_col]].copy()
            df_agg.columns = [region_col, value_col]

        if not value_col or value_col not in df_agg.columns:
            raise ValueError(f"数值列 {value_col} 不存在或无法聚合")

        # 过滤掉映射失败的地名
        valid_rows = []
        skipped_count = 0
        for _, row in df_agg.iterrows():
            raw_name = str(row[region_col])
            geo_name = _to_geo_name(raw_name)
            if geo_name in _PROVINCE_CENTROIDS or _GEO_PROVINCE_NAMES.get(geo_name):
                valid_rows.append(row)
            else:
                skipped_count += 1
        if skipped_count > 0 and valid_rows:
            import logging
            skipped_names = [str(row[region_col]) for _, row in df_agg.iterrows()
                             if _to_geo_name(str(row[region_col])) not in _PROVINCE_CENTROIDS
                             and _GEO_PROVINCE_NAMES.get(_to_geo_name(str(row[region_col]))) is None]
            logging.getLogger(__name__).warning(
                f"地图过滤了 {skipped_count} 条无法匹配 GeoJSON 的地名"
                f"（如：{', '.join(n[:8] for n in skipped_names[:5])}）"
            )
            df_agg = pd.DataFrame(valid_rows)
        if df_agg.empty:
            raise ValueError(
                f"列「{region_col}」中的值（如 {', '.join(sample_names[:5])}）"
                f"无法匹配中国地图省份名。请尝试用「省份」列作为 X 轴。"
            )

    # 确保列存在
    if not value_col or value_col not in df_agg.columns:
        raise ValueError(f"数值列 {value_col} 不存在或无法聚合")

    max_val = float(df_agg[value_col].max())
    min_val = float(df_agg[value_col].min())

    regions = []
    scatter_data = []

    for _, row in df_agg.iterrows():
        if is_region_mode:
            # 已展开为 province_geo / value_col，直接从 df_agg 取值
            geo_name = str(row['province_geo'])
            val = float(row[value_col])
            rname = province_region_map.get(geo_name, '')
        else:
            raw_name = str(row[region_col])
            geo_name = _to_geo_name(raw_name)
            val = float(row[value_col])
            rname = ''

        color = _color_by_value(val, min_val, max_val)

        # 标签：region 模式显示地区名，否则显示省份名
        label_text = rname if is_region_mode and rname else geo_name
        regions.append({
            "name": geo_name,
            "itemStyle": {"areaColor": color},
            "label": {
                "show": True,
                "color": "#7DD3FC",
                "fontSize": 11,
            },
        })

        # 散点：region 模式用大区中心，省份模式用省份中心
        if is_region_mode and rname:
            centroid = _REGION_CENTROIDS.get(rname)
            point_name = rname
        else:
            centroid = _PROVINCE_CENTROIDS.get(geo_name)
            point_name = geo_name

        if centroid:
            scatter_data.append({
                "name": point_name,
                "value": [*centroid, val],
            })

    # region 模式下去重散点（同一大区的省份散点在同一坐标）
    if is_region_mode and scatter_data:
        _seen = {}
        _deduped = []
        for pt in scatter_data:
            key = pt["name"]
            if key not in _seen:
                _seen[key] = True
                _deduped.append(pt)
        scatter_data = _deduped

    # ★ 构建 region mode 下的 tooltip formatter：显示「地区名 → 省份名」
    _fmt_base = (title or value_col)
    if is_region_mode and province_region_map:
        tooltip_fmt = (
            "function(p) {"
            "  var pm = " + json.dumps(province_region_map, ensure_ascii=False) + ";"
            "  var r = pm[p.name] || '';"
            "  return '<b>' + p.name + '</b>"
            "        + (r ? '（' + r + '）' : '')"
            "        + '<br/>" + _fmt_base + ": ' + p.value[2];"
            "}"
        )
    else:
        tooltip_fmt = "{b}<br/>" + _fmt_base + ": {c}"

    # ★ region mode：隐藏省份标签，只显示大区名
    if is_region_mode:
        for r in regions:
            r.setdefault("label", {})["show"] = False

    result = {
        "backgroundColor": "transparent",
        "title": {
            "text": title,
            "left": "center",
            "top": 8,
            "textStyle": {"color": "#7DD3FC", "fontSize": 18, "fontWeight": "bold",
                          "textShadowBlur": 10, "textShadowColor": "rgba(59,130,246,0.5)"},
        },
        "tooltip": {
            "trigger": "item",
            "backgroundColor": "rgba(15,12,41,0.95)",
            "borderColor": "#38BDF8",
            "borderWidth": 1,
            "textStyle": {"color": "#F8FAFC", "fontSize": 13},
            "formatter": tooltip_fmt,
        },
        "visualMap": {
            "show": True,
            "text": ["高", "低"],
            "min": min_val,
            "max": max_val,
            "calculable": False,
            "inRange": {"color": ["#13243F", "#0c4a6e", "#0369a1", "#0ea5e9", "#38BDF8", "#67E8F9", "#22D3EE"]},
            "textStyle": {"color": "#7DD3FC"},
            "orient": "horizontal",
            "left": "center",
            "bottom": 10,
        },
        "geo": {
            "map": "china",
            "roam": True,
            "zoom": 1.15,
            "center": [104.5, 36],
            "aspectScale": 0.85,
            "regions": regions,
            "itemStyle": {
                "areaColor": "#23304E",
                "borderColor": "rgba(255,255,255,0.08)",
                "borderWidth": 1,
                "shadowBlur": 6,
                "shadowColor": "rgba(56,189,248,0.25)",
            },
            "emphasis": {
                "itemStyle": {
                    "areaColor": "#38BDF8",
                    "shadowBlur": 25,
                    "shadowColor": "rgba(56,189,248,0.7)",
                },
                "label": {
                    "show": True,
                    "color": "#F8FAFC",
                    "fontSize": 14,
                    "fontWeight": "bold",
                    "textShadowBlur": 8,
                    "textShadowColor": "rgba(56,189,248,0.8)",
                },
            },
        },
        "series": [
            {
                "type": "effectScatter",
                "coordinateSystem": "geo",
                "data": scatter_data,
                "symbol": "circle",
                "symbolSize": 6,
                "showEffectOn": "render",
                "rippleEffect": {
                    "brushType": "stroke",
                    "scale": 4,
                    "period": 4,
                    "color": "#7DD3FC",
                },
                "itemStyle": {"color": "#F8FAFC", "shadowBlur": 10, "shadowColor": "rgba(56,189,248,0.8)"},
                "label": {
                    "show": True,
                    "position": "top",
                    "distance": 10,
                    "color": "#7DD3FC",
                    "fontSize": 11,
                    "fontWeight": "bold",
                    "formatter": "{c}",
                    "textShadowBlur": 6,
                    "textShadowColor": "rgba(56,189,248,0.6)",
                },
                "emphasis": {
                    "scale": 2,
                    "itemStyle": {"color": "#F8FAFC", "shadowBlur": 20, "shadowColor": "rgba(56,189,248,0.9)"},
                    "label": {"fontSize": 15, "color": "#F8FAFC"},
                },
            },
        ],
    }

    # ★ region mode 覆盖：隐藏省份标签，突出大区名称
    if is_region_mode:
        # 隐藏所有省份标签（包括 hover 时）
        result["geo"]["label"] = {"show": False}
        result["geo"]["emphasis"]["label"]["show"] = False
        # 放大散点并显示大区名 + 数值
        result["series"][0]["symbolSize"] = 14
        result["series"][0]["label"].update({
            "show": True,
            "fontSize": 15,
            "fontWeight": "bold",
            "color": "#F8FAFC",
            "formatter": "{b}\n{c}",
            "textShadowBlur": 10,
            "textShadowColor": "rgba(56,189,248,0.9)",
        })

    return result


# ==================== 降采样工具函数 ====================

def _downsample_indices(n: int, max_n: int) -> list:
    """返回 [0, n) 内均匀分布的 <=max_n 个下标（含首尾，按 np.linspace 取整去重）。"""
    if n <= max_n:
        return list(range(n))
    indices = np.linspace(0, n - 1, max_n, dtype=int)
    return sorted(set(int(i) for i in indices))


def _cap_heatmap(option: dict, series_list: list) -> dict:
    """热力图专用护栏：不再套用一维类目轴截断。

    - cohort（同期群下三角矩阵）：完整显示，不做任何截断（前端按尺寸撑高）。
    - correlation（相关性）：类目超上限时，对 x/y 轴均匀降采样，并**正确重映射**
      series.data 的坐标 [xi, yi, v]，替代旧逻辑的坐标错位过滤。
    """
    kind = option.get("_heatmap_kind", "correlation")
    if kind == "cohort":
        return option

    # correlation：先算好 x/y 两轴各自的 旧->新 下标映射，再统一重映射坐标
    remaps = {}
    for axis_key in ("xAxis", "yAxis"):
        ax = option.get(axis_key)
        a = ax if isinstance(ax, dict) else (ax[0] if isinstance(ax, list) and ax and isinstance(ax[0], dict) else None)
        if not isinstance(a, dict):
            continue
        ax_data = a.get("data")
        if not isinstance(ax_data, list) or len(ax_data) <= _MAX_CATEGORY:
            continue
        keep_idxs = _downsample_indices(len(ax_data), _MAX_CATEGORY)
        remaps[axis_key] = {old: new for new, old in enumerate(keep_idxs)}
        a["data"] = [ax_data[i] for i in keep_idxs]

    if remaps:
        rx = remaps.get("xAxis")
        ry = remaps.get("yAxis")
        for s in series_list:
            if str(s.get("type")) != "heatmap":
                continue
            hd = s.get("data", [])
            if not isinstance(hd, list):
                continue
            new_data = []
            for d in hd:
                if not (isinstance(d, (list, tuple)) and len(d) >= 2):
                    continue
                xi, yi = d[0], d[1]
                if rx and xi not in rx:
                    continue
                if ry and yi not in ry:
                    continue
                nx = rx.get(xi, xi) if rx else xi
                ny = ry.get(yi, yi) if ry else yi
                new_data.append([nx, ny, d[2]])
            s["data"] = new_data
    return option


def _cap_option_data(option: dict) -> dict:
    """统一护栏：对类目轴 / 数值散点 / 饼图 / 热力图超大序列做均匀降采样。
    仅在超出阈值时原地修改 option，小数据零影响。
    """
    if not option:
        return option

    series_list = option.get("series", [])
    if not series_list:
        return option

    # 热力图走专用护栏，避免被一维类目轴截断误伤
    if any(str(s.get("type")) == "heatmap" for s in series_list):
        return _cap_heatmap(option, series_list)

    # ---- 1. 类目轴图表（bar / line / area / histogram / waterfall / stacked_bar） ----
    x_axis = option.get("xAxis")
    if isinstance(x_axis, dict):
        x_data = x_axis.get("data")
    elif isinstance(x_axis, list) and len(x_axis) > 0:
        x_data = x_axis[0].get("data") if isinstance(x_axis[0], dict) else None
    else:
        x_data = None

    if x_data and isinstance(x_data, list) and len(x_data) > _MAX_CATEGORY:
        keep_idxs = _downsample_indices(len(x_data), _MAX_CATEGORY)
        # 裁剪 xAxis.data
        x_axis["data"] = [x_data[i] for i in keep_idxs]
        # 同步裁剪每个 series.data
        for s in series_list:
            s_data = s.get("data")
            if isinstance(s_data, list):
                s["data"] = [s_data[i] for i in keep_idxs if i < len(s_data)]
        return option

    # ---- 2. 数值轴散点/气泡（data 为 [x,y] 或 [x,y,size] 数组） ----
    #    无 xAxis.data 但有系列 data 超过上限 → 均匀采样
    for s in series_list:
        s_type = str(s.get("type", ""))
        s_data = s.get("data", [])
        if not isinstance(s_data, list) or len(s_data) <= _MAX_SERIES_POINTS:
            continue
        # 只对散点/气泡类做采样（折线/柱状的数值轴情况 → 通常类目轴已处理，此处仅保护）
        if s_type in ("scatter", "bubble", "effectScatter"):
            keep_idxs = _downsample_indices(len(s_data), _MAX_SERIES_POINTS)
            s["data"] = [s_data[i] for i in keep_idxs]
        elif s_type in ("bar", "line", "area"):
            # 无类目轴的数值型 → 仍需采样
            keep_idxs = _downsample_indices(len(s_data), _MAX_SERIES_POINTS)
            s["data"] = [s_data[i] for i in keep_idxs]
        # 箱线图 series.data 通常很小，跳过
        # 其他类型：仅当 data 是纯数组时采样
        elif all(isinstance(d, (int, float)) for d in s_data[:10] if d is not None):
            keep_idxs = _downsample_indices(len(s_data), _MAX_SERIES_POINTS)
            s["data"] = [s_data[i] for i in keep_idxs]

    # ---- 3. 饼图 / 树图 / 词云：Top N + "其他" ----
    for s in series_list:
        s_type = str(s.get("type", ""))
        if s_type not in ("pie", "treemap"):
            continue
        s_data = s.get("data", [])
        if not isinstance(s_data, list) or len(s_data) <= _MAX_PIE_SLICES:
            continue
        # 按 value 降序取 Top N
        def _val(d):
            if isinstance(d, dict):
                return float(d.get("value", 0) or 0)
            return float(d) if isinstance(d, (int, float)) else 0
        sorted_data = sorted(s_data, key=_val, reverse=True)
        top = sorted_data[:_MAX_PIE_SLICES - 1]
        rest = sorted_data[_MAX_PIE_SLICES - 1:]
        other_val = sum(_val(d) for d in rest)
        if other_val > 0:
            top.append({"name": "其他(合计)", "value": other_val,
                        "itemStyle": {"color": "rgba(255,255,255,0.08)"}})
        s["data"] = top

    return option


# ==================== 同期群专用图表（自包含，不委托他函数） ====================
# 未观测格哨兵值（须与 src/analysis_engine/models/cohort.py 的 SENTINEL 一致）
COHORT_SENTINEL = -1.0


def create_cohort_heatmap(df, x, y, title="同期群热力图", **ignored):
    """下三角热力图（留存率 / 客单价 / 净毛利）。

    数据行键：Index_j(x 轴=月偏移), 首单月(y 轴=cohort), value, ci_lower/ci_upper(可选)。
    未观测格用 COHORT_SENTINEL(-1) + visualMap 分段灰显(#334155)。
    """
    try:
        if df is None or df.empty or x not in df.columns or y not in df.columns:
            return {}
        xvals = sorted(df[x].dropna().unique().tolist())
        yvals = sorted(df[y].dropna().unique().tolist())

        # 强制窗口截断：无论上游打包阶段是否生效，渲染端做最后兜底。
        # 横轴 j 限制 _COHORT_WINDOW_MONTHS，纵轴取最近 N 个 cohort。
        _W = _COHORT_WINDOW_MONTHS
        xvals = [v for v in xvals if v < _W]
        yvals = yvals[-_W:] if len(yvals) > _W else yvals

        xidx = {v: i for i, v in enumerate(xvals)}
        yidx = {v: i for i, v in enumerate(yvals)}
        data = []
        for _, r in df.iterrows():
            xi = xidx.get(r[x])
            yi = yidx.get(r[y])
            if xi is None or yi is None:
                continue
            try:
                v = float(r.get("value", COHORT_SENTINEL))
            except (TypeError, ValueError):
                v = COHORT_SENTINEL
            data.append([xi, yi, v])

        obs = [d[2] for d in data if d[2] != COHORT_SENTINEL]
        vmin = min(obs) if obs else 0.0
        vmax = max(obs) if obs else 1.0
        if vmax - vmin < 1e-9:
            vmax = vmin + 1.0

        pieces = [{"value": COHORT_SENTINEL, "label": "未观测", "color": "#334155"}]
        n = 5
        step = (vmax - vmin) / n
        for k in range(n):
            lo = vmin + k * step
            if k < n - 1:
                hi = vmin + (k + 1) * step
                pieces.append({"gte": lo, "lt": hi, "color": BLUE_PALETTE[k % len(BLUE_PALETTE)]})
            else:
                pieces.append({"gte": lo, "lte": vmax, "color": BLUE_PALETTE[k % len(BLUE_PALETTE)]})

        x_axis = {
            "type": "category",
            "data": [str(v) for v in xvals],
            "name": "月偏移 j",
            "nameLocation": "middle",
            "nameGap": 30,
            "axisLine": {"color": "rgba(255,255,255,0.08)"},
        }
        y_axis = {
            "type": "category",
            "data": [str(v) for v in yvals],
            "inverse": True,
            "axisLine": {"color": "rgba(255,255,255,0.08)"},
        }
        visual_map = {
            "type": "piecewise",
            "text": ["高", "低"],
            "pieces": pieces,
            "orient": "horizontal",
            "left": "center",
            "bottom": 0,
            "textStyle": {"color": GALAXY["text_secondary"]},
        }
        series = [{
            "type": "heatmap",
            "data": data,
            "label": {"show": False},
            "emphasis": {"itemStyle": {"borderColor": "#FFFFFF", "borderWidth": 1}},
        }]
        option = dict(_get_default_title(title))
        option["tooltip"] = DARK_THEME["tooltip"]
        option["grid"] = {"top": 60, "left": 120, "right": 30, "bottom": 70}
        option["xAxis"] = x_axis
        option["yAxis"] = y_axis
        option["visualMap"] = visual_map
        option["series"] = series
        option["_heatmap_kind"] = "cohort"  # 同期群下三角矩阵：护栏须完整显示
        return option
    except Exception:
        return {}


def create_cohort_stacked(df, x, y, title="分组堆叠条形图", **ignored):
    """渠道/类目分组堆叠条形图（自包含，跳过 _auto_groupby 避免加总破坏）。

    数据行键：x(首单月), group(维度值), value(指标)。
    """
    try:
        if df is None or df.empty or x not in df.columns or y not in df.columns:
            return {}
        group_col = "group"
        if group_col not in df.columns:
            return {}
        xcats = sorted(df[x].dropna().unique().tolist())
        _W = _COHORT_WINDOW_MONTHS
        if len(xcats) > _W:
            xcats = xcats[-_W:]  # 首单月取最近 18 个
        groups = list(df[group_col].dropna().unique().tolist())
        xidx = {v: i for i, v in enumerate(xcats)}
        series = []
        for gi, g in enumerate(groups):
            sub = df[df[group_col] == g]
            ymap = {}
            for _, r in sub.iterrows():
                xi = xidx.get(r[x])
                if xi is None:
                    continue
                ymap[xi] = r.get(y, 0)
            data = [ymap.get(i, 0) for i in range(len(xcats))]
            series.append({
                "name": str(g),
                "type": "bar",
                "stack": "total",
                "data": data,
                "itemStyle": {"color": BLUE_PALETTE[gi % len(BLUE_PALETTE)]},
            })
        option = dict(_get_default_title(title))
        option["tooltip"] = {**DARK_THEME["tooltip"], "trigger": "axis", "axisPointer": {"type": "shadow"}}
        option["legend"] = DARK_THEME["legend"]
        option["toolbox"] = DARK_THEME["toolbox"]
        option["grid"] = {"top": 60, "left": 60, "right": 30, "bottom": 60}
        option["xAxis"] = {
            "type": "category",
            "data": [str(v) for v in xcats],
            "axisLabel": {"rotate": 30 if len(xcats) > 8 else 0},
            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}},
        }
        option["yAxis"] = {"type": "value",
                             "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
        option["series"] = series
        return option
    except Exception:
        return {}


def _build_cohort_line_option(df, x, y, title="分组折线图", subtext="", truncate=True, show_title=True):
    """分组折线图内部构建器（x=类目轴, group 每值一条折线）。

    参数：
      subtext    : 副标题（仅 show_title=True 时写入 option["title"]["subtext"]）。
      truncate   : True=类目轴取最近 _COHORT_WINDOW_MONTHS 个（基座纵向口径）；
                   False=保留完整时间线（横截面健康度口径，不截断）。
      show_title : True=写入 title/subtext（基座纵向需要）；
                   False=option 不写 title 字段，ECharts 不渲染标题区与副标题（横截面健康度口径，标题在容器外层给出）。
    数据行键：x(类目), group(维度值), value(指标)。
    """
    try:
        if df is None or df.empty or x not in df.columns or y not in df.columns:
            return {}
        group_col = "group"
        if group_col not in df.columns:
            return {}
        xcats = sorted(df[x].dropna().unique().tolist())
        if truncate:
            _W = _COHORT_WINDOW_MONTHS
            if len(xcats) > _W:
                xcats = xcats[-_W:]  # 首单月取最近 18 个
        groups = list(df[group_col].dropna().unique().tolist())
        xidx = {v: i for i, v in enumerate(xcats)}
        series = []
        for gi, g in enumerate(groups):
            sub = df[df[group_col] == g]
            ymap = {}
            for _, r in sub.iterrows():
                xi = xidx.get(r[x])
                if xi is None:
                    continue
                ymap[xi] = r.get(y, 0)
            data = [ymap.get(i, None) for i in range(len(xcats))]
            series.append({
                "name": str(g),
                "type": "line",
                "connectNulls": True,
                "data": data,
                "emphasis": {"focus": "series"},
                "lineStyle": {"width": 2},
                "itemStyle": {"color": BLUE_PALETTE[gi % len(BLUE_PALETTE)]},
            })
        option = dict(DARK_THEME)
        if show_title:
            option["title"] = _get_default_title(title)
            if subtext:
                option["title"]["subtext"] = subtext
                option["title"]["top"] = 24
        # show_title=False 时不写 option["title"]，ECharts 不会渲染标题区与副标题
        option["tooltip"] = {**DARK_THEME["tooltip"], "trigger": "axis"}
        option["legend"] = DARK_THEME["legend"]
        option["toolbox"] = DARK_THEME["toolbox"]
        option["grid"] = {"top": 60, "left": 60, "right": 30, "bottom": 60}
        option["xAxis"] = {
            "type": "category",
            "data": [str(v) for v in xcats],
            "boundaryGap": False,
            "axisLabel": {"rotate": 30 if len(xcats) > 8 else 0},
            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}},
        }
        option["yAxis"] = {"type": "value",
                             "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
        option["series"] = series
        return option
    except Exception:
        return {}


def create_cohort_line(df, x, y, title="分组折线图", **ignored):
    """渠道/类目分组折线图（自包含，跳过 _auto_groupby 避免加总破坏）。

    与 create_cohort_stacked 同源：x=首单月（category 轴），每个 group 一条折线，
    不堆叠。用于「各渠道/类目 M1 留存用户数」等需横向对比各群走势的场景。
    默认截断首单月至最近窗口（纵向口径）。
    """
    return _build_cohort_line_option(df, x, y, title=title, subtext="", truncate=True)


def create_cohort_active_line(df, x, y, title="月度活跃留存用户数", **ignored):
    """横截面健康度：月度活跃留存用户数折线（ALL 总览 + 分维度）。

    与基座纵向同期群留存率（按首单月追人、j=1 次月留存）口径严格区分：
    这里是横截面——锚点=当月，当月有订单且首购月 < 当月的老客即计，无「次月」约束。
    标题/副标题在容器外层给出（A 分支折线图卡片自带「横截面健康度」语义），ECharts option 内不再渲染。
    """
    return _build_cohort_line_option(
        df, x, y, title=title,
        subtext="",
        truncate=False,
        show_title=False,
    )


def create_cohort_trend(df, x, y, title="同期群质量趋势", **ignored):
    """偏移 j 留存趋势折线（每 j 一条线，显著性 '*' 以 markPoint 标注）。

    数据行键：x(首单月), Index_j(序列分组), 留存率, mark(可选 '*')。
    """
    try:
        if df is None or df.empty or x not in df.columns or y not in df.columns:
            return {}
        jcol = "Index_j"
        if jcol not in df.columns:
            return {}
        xcats = sorted(df[x].dropna().unique().tolist())
        _W = _COHORT_WINDOW_MONTHS
        if len(xcats) > _W:
            xcats = xcats[-_W:]  # 首单月取最近 18 个
        xidx = {v: i for i, v in enumerate(xcats)}
        js = sorted(df[jcol].dropna().unique().tolist())
        if len(js) > _W:
            js = js[:_W]  # Index_j 取前 18 条折线
        series = []
        for j in js:
            sub = df[df[jcol] == j]
            ymap = {}
            marks = {}
            for _, r in sub.iterrows():
                xi = xidx.get(r[x])
                if xi is None:
                    continue
                ymap[xi] = r.get(y, 0)
                mk = r.get("mark", "")
                if mk:
                    marks[xi] = mk
            data = [ymap.get(i, None) for i in range(len(xcats))]
            s = {
                "name": "j=%s" % j,
                "type": "line",
                "data": data,
                "connectNulls": True,
                "itemStyle": {"color": BLUE_PALETTE[(int(j) - 1) % len(BLUE_PALETTE)]},
                "lineStyle": {"width": 2},
            }
            if marks:
                s["markPoint"] = {
                    "symbol": "pin",
                    "symbolSize": 38,
                    "data": [{"coord": [xi, ymap[xi]], "value": "*"} for xi in marks if xi in ymap],
                    "itemStyle": {"color": GALAXY["danger"]},
                    "label": {"color": "#FFFFFF", "fontSize": 12},
                }
            series.append(s)
        option = dict(_get_default_title(title))
        option["tooltip"] = {**DARK_THEME["tooltip"], "trigger": "axis"}
        option["legend"] = DARK_THEME["legend"]
        option["toolbox"] = DARK_THEME["toolbox"]
        option["grid"] = {"top": 60, "left": 60, "right": 30, "bottom": 50}
        option["xAxis"] = {
            "type": "category",
            "data": [str(v) for v in xcats],
            "axisLabel": {"rotate": 30 if len(xcats) > 8 else 0},
            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}},
        }
        option["yAxis"] = {"type": "value",
                             "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
        option["series"] = series
        return option
    except Exception:
        return {}


def create_dual_axis(df, x, y, title="净GMV vs 净毛利",
                     right_col=None, **ignored):
    """双轴图：左轴折线，右轴柱状。

    数据行键：x(类目), y(左轴列), right_col(右轴列，默认"净毛利")。
    """
    try:
        if df is None or df.empty or x not in df.columns or y not in df.columns:
            return {}
        right_col = right_col or "净毛利"
        if right_col not in df.columns:
            return {}
        xcats = sorted(df[x].dropna().unique().tolist())
        xidx = {v: i for i, v in enumerate(xcats)}
        ymap = {}
        rmap = {}
        for _, r in df.iterrows():
            xi = xidx.get(r[x])
            if xi is None:
                continue
            ymap[xi] = r.get(y, 0)
            rmap[xi] = r.get(right_col, 0)
        left = [ymap.get(i, 0) for i in range(len(xcats))]
        right = [rmap.get(i, 0) for i in range(len(xcats))]
        option = dict(_get_default_title(title))
        option["tooltip"] = {**DARK_THEME["tooltip"], "trigger": "axis"}
        option["legend"] = DARK_THEME["legend"]
        option["toolbox"] = DARK_THEME["toolbox"]
        option["grid"] = {"top": 60, "left": 80, "right": 80, "bottom": 50}
        option["xAxis"] = {
            "type": "category",
            "data": [str(v) for v in xcats],
            "axisLabel": {"rotate": 30 if len(xcats) > 8 else 0},
            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}},
        }
        option["yAxis"] = [
            {"type": "value", "name": y,
             "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
            {"type": "value", "name": right_col,
             "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}},
        ]
        option["series"] = [
            {"name": y, "type": "line", "yAxisIndex": 0, "data": left,
             "itemStyle": {"color": BLUE_PALETTE[0]}, "lineStyle": {"width": 2}},
            {"name": right_col, "type": "bar", "yAxisIndex": 1, "data": right,
             "itemStyle": {"color": BLUE_PALETTE[2]}},
        ]
        return option
    except Exception:
        return {}


def create_ranking_chart(df: pd.DataFrame, x: str, y: str,
                         title: str = "排名排行", **ignored) -> Dict[str, Any]:
    """水平条形排行图：Y 轴类别型（排名/名称，TOP1 在最上方）+ X 轴数值（个体价值）。

    入参 df：含 x=排名标签列、y=数值列；可选含「用户ID」列用于组合标签。
    适用于 Top N 排行场景（如 CLV Top15 客户生命周期价值排行）。
    """
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return {}
    labels = df[x].astype(str).tolist()
    vals = pd.to_numeric(df[y], errors="coerce").fillna(0.0).tolist()
    # 含「用户ID」列时 y 轴只显示用户ID（去掉「排名 ·」前缀冗余）
    if "用户ID" in df.columns:
        labels = df["用户ID"].astype(str).tolist()
    if len(vals) == 0:
        return {}

    option = {
        "title": _get_default_title(title),
        "tooltip": {**DARK_THEME["tooltip"], "trigger": "axis",
                    "axisPointer": {"type": "shadow"}},
        "toolbox": DARK_THEME["toolbox"],
        "grid": {"top": 60, "left": 120, "right": 80, "bottom": 40},
        "xAxis": {
            "type": "value",
            "name": "客户生命周期价值（¥）",
            "nameLocation": "middle", "nameGap": 30,
            "nameTextStyle": {"color": GALAXY["text_primary"], "fontWeight": "bold"},
            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}},
            "axisLabel": {"color": GALAXY["text_primary"]},
            "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.06)"}},
        },
        "yAxis": {
            "type": "category",
            "data": labels,
            "inverse": True,                       # TOP1 固定在最上方
            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}},
            "axisLabel": {"color": GALAXY["text_primary"]},
            "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.06)"}},
        },
        "series": [{
            "name": "客户生命周期价值",
            "type": "bar",
            "data": [{"value": float(v),
                      "itemStyle": {"color": BLUE_PALETTE[0]}} for v in vals],
            "barMaxWidth": 22,
            "itemStyle": {"color": BLUE_PALETTE[0], "borderRadius": [0, 4, 4, 0]},
            "label": {
                "show": True,
                "position": "right",
                "formatter": "{c}",
                "color": GALAXY["text_primary"],
                "fontSize": 11,
            },
        }],
    }
    return option


# ==================== RFM 专用图表（自包含，不依赖额外 kwargs） ====================

def create_rfm_line(df, x=None, y=None, title="分层占比趋势", **ignored):
    """折线图：x=月份 / y=占比 / series=分层（每分层一条 line）。

    数据行键：x(月份标签), y(数值占比 0~1), series(分层名)。
    """
    try:
        if df is None or df.empty or "x" not in df.columns or "y" not in df.columns or "series" not in df.columns:
            return {}
        months = sorted(df["x"].dropna().unique().tolist())
        midx = {m: i for i, m in enumerate(months)}
        segs = sorted(df["series"].dropna().unique().tolist())
        series = []
        for i, seg in enumerate(segs):
            sub = df[df["series"] == seg]
            vmap = {}
            for _, r in sub.iterrows():
                if r["x"] in midx:
                    try:
                        vmap[midx[r["x"]]] = float(r["y"])
                    except (TypeError, ValueError):
                        continue
            data = [round(vmap.get(i, 0.0) * 100, 2) for i in range(len(months))]
            series.append({
                "name": str(seg),
                "type": "line",
                "smooth": True,
                "data": data,
                "itemStyle": {"color": BLUE_PALETTE[i % len(BLUE_PALETTE)]},
                "lineStyle": {"width": 2},
            })
        option = dict(DARK_THEME)
        option["title"] = _get_default_title(title)
        option["tooltip"] = {**DARK_THEME["tooltip"], "trigger": "axis"}
        option["legend"] = {**DARK_THEME["legend"], "type": "scroll"}
        option["grid"] = {"top": 60, "left": 70, "right": 40, "bottom": 70}
        option["xAxis"] = {"type": "category", "data": [str(m) for m in months],
                            "axisLabel": {"rotate": 30 if len(months) > 8 else 0},
                            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
        option["yAxis"] = {"type": "value", "name": "占比(%)",
                            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
        option["series"] = series
        return option
    except Exception:
        return {}


def create_heatmap_2d(df, x=None, y=None, title="分层 × 地域 净毛利均值", **ignored):
    """二维热力图：x/y 为类目轴，value 为数值（不走相关性分支）。

    数据行键：x(类目1/分层), y(类目2/地域), value(数值)。
    必须置 _heatmap_kind='cohort'，规避 _cap_heatmap 相关性分支的轴降采样。
    """
    try:
        if df is None or df.empty or "x" not in df.columns or "y" not in df.columns or "value" not in df.columns:
            return {}
        xcats = sorted(df["x"].dropna().unique().tolist())
        ycats = sorted(df["y"].dropna().unique().tolist())
        xidx = {v: i for i, v in enumerate(xcats)}
        yidx = {v: i for i, v in enumerate(ycats)}
        data = []
        vals = []
        for _, r in df.iterrows():
            xi = xidx.get(r["x"])
            yi = yidx.get(r["y"])
            try:
                v = float(r["value"])
            except (TypeError, ValueError):
                continue
            if xi is None or yi is None:
                continue
            data.append([xi, yi, v])
            vals.append(v)
        if not data:
            return {}
        vmin = min(vals)
        vmax = max(vals)
        if vmax - vmin < 1e-9:
            vmax = vmin + 1.0
        grad = [GALAXY["card_bg"], BLUE_PALETTE[2], BLUE_PALETTE[0],
                BLUE_PALETTE[3], BLUE_PALETTE[5]]
        option = dict(DARK_THEME)
        option["title"] = _get_default_title(title)
        option["tooltip"] = {**DARK_THEME["tooltip"], "trigger": "item",
                              "formatter": "{c[2]}"}
        option["grid"] = {"top": 60, "left": 90, "right": 30, "bottom": 60}
        option["xAxis"] = {"type": "category", "data": [str(v) for v in xcats],
                            "axisLabel": {"rotate": 30 if len(xcats) > 8 else 0},
                            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
        option["yAxis"] = {"type": "category", "data": [str(v) for v in ycats],
                            "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.08)"}}}
        option["visualMap"] = {
            "min": round(vmin, 2), "max": round(vmax, 2),
            "calculable": True, "orient": "horizontal", "left": "center", "bottom": 10,
            "textStyle": {"color": GALAXY["text_secondary"]},
            "inRange": {"color": grad},
        }
        option["series"] = [{
            "type": "heatmap",
            "data": data,
            "label": {"show": False},
            "emphasis": {"itemStyle": {"shadowBlur": 8, "shadowColor": "rgba(56,189,248,0.6)"}},
        }]
        # 关键：走 cohort 护栏分支，不被相关性分支截断
        option["_heatmap_kind"] = "cohort"
        return option
    except Exception:
        return {}


def create_sankey(df, x=None, y=None, title="群体流转桑基图", **ignored):
    """桑基图：source→target 的流量（value=人数）。

    数据行键：source(分层), target(分层), value(人数)。
    过滤 source==target 自环（ECharts 桑基图自环会布局异常）；
    若过滤后无 link（相邻期无迁移）→ 返回 {}（由调用方跳过）。
    """
    try:
        if df is None or df.empty or "source" not in df.columns or "target" not in df.columns or "value" not in df.columns:
            return {}
        nodes = []
        seen = set()
        for col in ("source", "target"):
            for v in df[col].dropna().unique().tolist():
                if v not in seen:
                    seen.add(v)
                    nodes.append({"name": str(v)})
        links = []
        stable = 0.0
        for _, r in df.iterrows():
            s = r["source"]
            t = r["target"]
            try:
                v = float(r["value"])
            except (TypeError, ValueError):
                continue
            if s == t:
                stable += v  # 自环（未迁移）不画，计入稳定用户量
                continue
            links.append({"source": str(s), "target": str(t), "value": v})
        if not links:
            return {}

        # 环检测：ECharts 桑基图本质是有向无环图(DAG)，若 link 图存在环
        # （用户在群体间来回流转，如 高价值核心客户→潜力高价值客户→高价值核心客户），整张图
        # 渲染失败、画布空白。用入度拓扑排序判定：遍历结束仍有节点未访问
        # 即存在环 → 返回 {}，由上层丢弃该卡片（画不出就不占位置）。
        indeg = {n["name"]: 0 for n in nodes}
        adj = {n["name"]: [] for n in nodes}
        for lk in links:
            s, t = lk["source"], lk["target"]
            if s in adj and t in adj:
                adj[s].append(t)
                indeg[t] += 1
        q = deque([name for name, d in indeg.items() if d == 0])
        visited = 0
        while q:
            u = q.popleft()
            visited += 1
            for w in adj[u]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)
        if visited < len(indeg):
            return {}

        option = dict(DARK_THEME)
        option["title"] = _get_default_title(title)
        option["tooltip"] = {**DARK_THEME["tooltip"], "trigger": "item"}
        option["series"] = [{
            "type": "sankey",
            "data": nodes,
            "links": links,
            "emphasis": {"focus": "adjacency"},
            "lineStyle": {"color": "gradient", "opacity": 0.55},
            "label": {"color": GALAXY["text_primary"], "fontSize": 11},
            "nodeWidth": 14,
            "nodeGap": 10,
        }]
        option["_stable_value"] = stable  # 稳定用户量（调用方可读，无害额外键）
        return option
    except Exception:
        return {}


def create_graph(df, x="source", y="target", title="商品关联网络图", **ignored):
    """关系网络图（graph）：节点=商品，连线=一起购买，线粗=共现次数，颜色=提升度。

    数据行键：source(商品A), target(商品B), value(共现次数), lift(提升度, 可选)。
    关联规则天生双向（A→B 与 B→A），调用方需已按 pair 去重为单边，避免重复连线。
    返回 ECharts graph option（力导向布局，节点大小=出现度，边粗细=共现次数）。
    """
    try:
        if df is None or len(df) == 0:
            return None
        # source/target 列名兼容：优先用传入 x/y，否则固定 source/target
        src_col = x if (x and x in df.columns) else ("source" if "source" in df.columns else None)
        tgt_col = y if (y and y in df.columns) else ("target" if "target" in df.columns else None)
        if src_col is None or tgt_col is None:
            return None
        # value 取除 source/target 外第一列；lift 取名为 'lift' 的列（可选）
        val_col = None
        for c in df.columns:
            if c not in (src_col, tgt_col):
                val_col = c
                break
        lift_col = "lift" if "lift" in df.columns else None

        links = []
        degree = {}
        for _, row in df.iterrows():
            s = str(row[src_col])
            t = str(row[tgt_col])
            try:
                v = float(row[val_col]) if val_col else 1.0
            except (TypeError, ValueError):
                v = 1.0
            try:
                lift = float(row[lift_col]) if lift_col else 1.0
            except (TypeError, ValueError):
                lift = 1.0
            links.append({"source": s, "target": t, "value": v, "lift": lift})
            degree[s] = degree.get(s, 0.0) + v
            degree[t] = degree.get(t, 0.0) + v
        if not degree:
            return None

        max_d = max(degree.values())
        min_d = min(degree.values())

        def node_size(d):
            if max_d == min_d:
                return 30
            return 20 + 34 * (d - min_d) / (max_d - min_d)

        counts = [lk["value"] for lk in links]
        max_c = max(counts) if counts else 1.0
        min_c = min(counts) if counts else 0.0

        def edge_style(v, lift):
            if max_c == min_c:
                width = 3.0
            else:
                width = 1.5 + 5.0 * (v - min_c) / (max_c - min_c)
            # 提升度 ≥1 为正向强关联（金色）；<1 为弱/负向（冷蓝）
            color = "#FBBF24" if lift >= 1.0 else "#60A5FA"
            return {"width": round(width, 2), "color": color, "opacity": 0.85, "curveness": 0.12}

        option = dict(DARK_THEME)
        option["title"] = _get_default_title(title)
        option["tooltip"] = {**DARK_THEME["tooltip"], "trigger": "item", "formatter": "{b}"}
        option["series"] = [{
            "type": "graph",
            "layout": "force",
            "roam": True,
            "label": {"show": True, "position": "right",
                      "color": GALAXY["text_primary"], "fontSize": 11},
            "force": {"repulsion": 160, "edgeLength": [60, 140], "gravity": 0.08},
            "data": [
                {"name": n, "symbolSize": node_size(d),
                 "value": round(d, 2),
                 "itemStyle": {"color": GALAXY["primary"]}}
                for n, d in degree.items()
            ],
            "links": [
                {"source": lk["source"], "target": lk["target"],
                 "value": round(lk["value"], 2),
                 "lineStyle": edge_style(lk["value"], lk["lift"])}
                for lk in links
            ],
            "emphasis": {"focus": "adjacency", "lineStyle": {"width": 7}},
            "lineStyle": {"color": "source", "curveness": 0.12},
            "itemStyle": {"color": GALAXY["primary"],
                          "borderColor": "rgba(255,255,255,0.18)"},
        }]
        return option
    except Exception:
        return None


def _infer_unit_hint(option: Dict[str, Any]) -> Optional[str]:
    """根据 y 轴数值范围返回合适的单位提示（纯数据，不生成 JS 代码）。
    返回 None 表示不需要单位格式化。"""
    series_list = option.get("series", [])
    if isinstance(series_list, dict):
        series_list = [series_list]
    max_v = 0
    for s in series_list:
        data = s.get("data", []) if isinstance(s, dict) else []
        for d in data:
            val = d.get("value", d) if isinstance(d, dict) else d
            if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
                max_v = max(max_v, abs(val))
    if max_v >= 100000000:
        return "亿"
    elif max_v >= 10000000:
        return "万"  # 千万级也用万
    elif max_v >= 10000:
        return "万"
    return None


# ==================== 统一入口 ====================

CHART_FUNCTIONS = {
    "bar": create_bar_chart,
    "hbar": create_horizontal_bar,
    "hbar_family": create_hbar_family,  # 图族：返回 {维度名: option}，前端遍历渲染
    "stacked_bar": create_stacked_bar,
    "line": create_line_chart,
    "area": create_area_chart,
    "scatter": create_scatter_chart,
    "bubble": create_bubble_chart,
    "bubble_matrix": create_bubble_matrix,
    "pie": create_pie_chart,
    "histogram": create_histogram,
    "box": create_box_plot,
    "heatmap": create_heatmap,
    "radar": create_radar_chart,
    "waterfall": create_waterfall,
    "treemap": create_treemap,
    "gl_map": create_gl_map,
    "map_3d": create_gl_map,  # 别名：数据洞察规则中"地区分布→3D地图"使用的类型名
    "cohort_heatmap": create_cohort_heatmap,
    "cohort_stacked": create_cohort_stacked,
    "cohort_line": create_cohort_line,
    "cohort_active_line": create_cohort_active_line,
    "cohort_trend": create_cohort_trend,
    "dual_axis": create_dual_axis,
    "ranking": create_ranking_chart,
    "funnel": create_funnel_chart,
    # ===== RFM 专用（自包含，从数据行固定列名读取参数） =====
    "rfm_line": create_rfm_line,
    "heatmap_2d": create_heatmap_2d,
    "sankey": create_sankey,
    "graph": create_graph,
}


def _parse_literal(value: Any) -> Optional[Any]:
    """若 value 是 JSON 数组字符串或已是 list/tuple，返回解析后的列表；否则返回 None。

    用于让 generate_chart 同时支持「传数据框列名」与「传现成数组字面量」两种用法。
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # 形如 ["a","b"] 或 [1,2,3] 或 [{"维度":..,"数值":..}]
        if s[0] in ("[", "（"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except (json.JSONDecodeError, ValueError):
                return None
    return None


def _coerce_literal_df(df: pd.DataFrame, chart_type: str, **kwargs) -> Optional[pd.DataFrame]:
    """当 LLM 直接传入现成数组字面量（而非列名）时，构造一张纯数据 DataFrame 供现有绘图函数使用。

    支持三种字面量形态：
      1. x=[...], y=[...]                      → 分类/数值列
      2. data=[{"维度":..,"数值":..}, ...]      → 维度/数值对（ranking/pie 友好）
      3. data=[{"name":..,"value":..}, ...]     → 别名形态
    返回 None 表示未检测到字面量（走原有「列名」路径）。
    """
    x_lit = _parse_literal(kwargs.get("x"))
    y_lit = _parse_literal(kwargs.get("y"))
    data_lit = _parse_literal(kwargs.get("data"))

    if x_lit is not None and y_lit is not None:
        # 长度不齐时以短为准截断
        n = min(len(x_lit), len(y_lit))
        return pd.DataFrame({"__x__": x_lit[:n], "__y__": y_lit[:n]})

    if data_lit is not None:
        rows = []
        for item in data_lit:
            if isinstance(item, dict):
                dim = item.get("维度") or item.get("name") or item.get("dim") or ""
                val = item.get("数值") or item.get("value") or item.get("val") or 0
                rows.append({"__x__": dim, "__y__": val})
            else:
                # 退化：单值序列，索引作为 x
                rows.append({"__x__": str(len(rows)), "__y__": item})
        if rows:
            return pd.DataFrame(rows)

    return None


def create_chart(df: pd.DataFrame, chart_type: str, **kwargs) -> Optional[Dict[str, Any]]:
    """统一 ECharts 图表创建入口，返回 ECharts option 字典（自动降采样护栏）"""
    if chart_type not in CHART_FUNCTIONS:
        raise ValueError(f"不支持的图表类型: {chart_type}。支持: {list(CHART_FUNCTIONS.keys())}")

    # ★ 字面量兼容层：若 LLM 直接传入现成数组（而非列名），构造纯数据 DataFrame 并改写 x/y
    literal_df = _coerce_literal_df(df, chart_type, **kwargs)
    if literal_df is not None:
        df = literal_df
        # 用构造出的内部列名覆盖 x/y，使下游绘图函数无需改动
        if kwargs.get("x") is not None:
            kwargs["x"] = "__x__"
        if kwargs.get("y") is not None:
            kwargs["y"] = "__y__"
        # ranking 缺省时也补上 x/y
        if "x" not in kwargs:
            kwargs["x"] = "__x__"
        if "y" not in kwargs:
            kwargs["y"] = "__y__"

    option = CHART_FUNCTIONS[chart_type](df, **kwargs)
    if option is not None:
        option = _cap_option_data(option)
        hint = _infer_unit_hint(option)
        if hint:
            option["_unitHint"] = hint
    return option
