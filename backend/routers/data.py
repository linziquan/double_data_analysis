"""
数据操作 API 路由（预览、列信息、分页等）
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import pandas as pd
import numpy as np
import io
import asyncio
import logging
import traceback
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor

from backend.services.session_manager import manager
from backend.utils.ai_error import enhance_ai_error
from src.data_loader import get_data_info, get_column_info
from src.utils.json_serializer import sanitize_json
from src.utils.helpers import get_numeric_columns, get_categorical_columns, get_datetime_columns
from config import QUOTA_BYTES

def _parse_missing_rate(row) -> float:
    """解析缺失率，兼容字符串 '0.0%' 和数字格式"""
    val = row.get("缺失率", row.get("missing_rate", 0))
    if isinstance(val, str):
        val = val.replace("%", "")
        try:
            return float(val) / 100.0
        except ValueError:
            return 0.0
    return float(val)


router = APIRouter()
logger = logging.getLogger(__name__)


class DataRequest(BaseModel):
    session_id: str


class PreviewRequest(DataRequest):
    rows: int = 100




def _safe_to_string(df_preview):
    """Safe DataFrame to string, avoiding GBK encoding issues on Windows."""
    import io
    buf = io.StringIO()
    df_preview.to_csv(buf, index=False, encoding='utf-8')
    return buf.getvalue()

@router.post("/data/preview")
async def data_preview(req: PreviewRequest):
    """获取数据预览"""
    df = manager.get_data(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="未找到数据，请先上传文件")
    preview = df.head(req.rows).replace({np.nan: None}).to_dict(orient="records")
    return sanitize_json({"success": True, "preview": preview, "total_rows": len(df)})




def _safe_to_string(df_preview):
    """Safe DataFrame to string, avoiding GBK encoding issues on Windows."""
    import io
    buf = io.StringIO()
    df_preview.to_csv(buf, index=False, encoding='utf-8')
    return buf.getvalue()

@router.post("/data/info")
async def data_info(req: DataRequest):
    """获取数据基本信息"""
    df = manager.get_data(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="未找到数据")
    info = get_data_info(df)
    return sanitize_json({"success": True, "info": info})




def _safe_to_string(df_preview):
    """Safe DataFrame to string, avoiding GBK encoding issues on Windows."""
    import io
    buf = io.StringIO()
    df_preview.to_csv(buf, index=False, encoding='utf-8')
    return buf.getvalue()

@router.post("/data/columns")
async def data_columns(req: DataRequest):
    """获取列信息"""
    df = manager.get_data(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="未找到数据")
    col_info = get_column_info(df)
    columns = []
    for _, row in col_info.iterrows():
        columns.append({
            "name": str(row.get("列名", "")),
            "dtype": str(row.get("数据类型", "")),
            "missing": int(row.get("缺失值数", 0)),
            "missing_rate": _parse_missing_rate(row),
            "unique": int(row.get("唯一值数", 0)),
            "sample": str(row.get("示例值", "")),
        })
    return sanitize_json({"success": True, "columns": columns})




def _safe_to_string(df_preview):
    """Safe DataFrame to string, avoiding GBK encoding issues on Windows."""
    import io
    buf = io.StringIO()
    df_preview.to_csv(buf, index=False, encoding='utf-8')
    return buf.getvalue()

@router.post("/data/column-types")
async def data_column_types(req: DataRequest):
    """获取各类列名"""
    df = manager.get_data(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="未找到数据")
    return sanitize_json({
        "success": True,
        "numeric_columns": get_numeric_columns(df),
        "categorical_columns": get_categorical_columns(df),
        "datetime_columns": get_datetime_columns(df),
        "all_columns": list(df.columns),
    })




def _safe_to_string(df_preview):
    """Safe DataFrame to string, avoiding GBK encoding issues on Windows."""
    import io
    buf = io.StringIO()
    df_preview.to_csv(buf, index=False, encoding='utf-8')
    return buf.getvalue()

@router.post("/data/summary")
async def data_summary(req: DataRequest):
    """获取数据摘要统计"""
    df = manager.get_data(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="未找到数据")
    summary = df.describe(include='all').to_dict()
    return sanitize_json({"success": True, "summary": summary})


class ComputeRequest(BaseModel):
    session_id: str
    query: str  # 用户的计算需求（自然语言）
    api_key: Optional[str] = ""
    base_url: Optional[str] = None
    model: Optional[str] = None




def _safe_to_string(df_preview):
    """Safe DataFrame to string, avoiding GBK encoding issues on Windows."""
    import io
    buf = io.StringIO()
    df_preview.to_csv(buf, index=False, encoding='utf-8')
    return buf.getvalue()

@router.post("/data/compute")
async def data_compute(req: ComputeRequest):
    """AI 数据计算：根据自然语言指令计算新列"""
    df = manager.get_data(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="未找到数据，请先上传文件")
    
    api_key = req.api_key or manager.get_api_key(req.session_id)
    if not api_key:
        raise HTTPException(status_code=400, detail="需要 API Key")
    
    # 获取原始列名
    original_columns = list(df.columns)
    
    # 构造数据上下文给 AI
    col_info = []
    for c in df.columns:
        col_info.append(f"  - {c} (类型: {df[c].dtype}), 示例值: {list(df[c].dropna().head(3))}")
    
    data_context = f"""数据列（共 {len(df)} 行）：
{chr(10).join(col_info)}

前 5 行数据：
{_safe_to_string(df.head(5))}"""

    prompt = f"""你是数据分析专家。用户需要你对数据做计算，新增计算列。

用户需求：{req.query}

当前数据：
{data_context}

请写一段 Python 代码，对 DataFrame（变量名 `df`）进行计算，新增列到 df 中。
要求：
1. 直接在 df 上新增列，不要创建新的 DataFrame
2. 只返回 Python 代码，用 ```python ... ``` 包裹
3. 代码中可以使用 pandas（已导入为 pd）和 numpy（已导入为 np）
4. 新列名用中文，简洁明了
5. 不要修改原有列的数据
6. 如果用户提到"同比"、"环比"，需要先确保有日期列并按日期排序
7. 如果用户提到"排名"、"占比"等，用 pandas 的 rank、transform 等方法
8. ★★★ 如果数据中有省份、地区、城市、部门等分类列，且用户需要计算汇总指标，必须先 groupby 该列再计算，例如：df['各省销售额'] = df.groupby('省份')['销售额'].transform('sum')
9. ★★★ 如果用户提到"各省份"、"各地区"、"按地区"、"按城市"等，必须用 groupby + transform/agg 计算，不要逐行简单运算

常见计算示例：
- 同比: df[f'销售额_同比'] = df.groupby(...)['销售额'].pct_change()
- 环比: df[f'销售额_环比'] = df['销售额'].pct_change()
- 累计: df[f'销售额_累计'] = df['销售额'].cumsum()
- 占比: df[f'销售额_占比'] = df['销售额'] / df['销售额'].sum()
- 排名: df[f'销售额_排名'] = df['销售额'].rank(ascending=False)
- 移动平均: df[f'销售额_MA7'] = df['销售额'].rolling(7).mean()
- ★ 各省汇总: df[f'各省销售额'] = df.groupby('省份')['销售额'].transform('sum')
- ★ 各省均值: df[f'各省平均销售额'] = df.groupby('省份')['销售额'].transform('mean')
- ★ 各省占比: df[f'各省销售额占比'] = df['销售额'] / df.groupby('省份')['销售额'].transform('sum')
- ★ 各省排名: df[f'省内排名'] = df.groupby('省份')['销售额'].rank(ascending=False)"""

    try:
        from src.ai_agent.agent import DataAnalysisAgent
        
        kwargs = {"api_key": api_key}
        if req.base_url:
            kwargs["base_url"] = req.base_url
        if req.model:
            kwargs["model"] = req.model
        agent = DataAnalysisAgent(**kwargs)
        
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = await loop.run_in_executor(
                executor,
                lambda: agent.client.chat.completions.create(
                    model=agent.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1024,
                )
            )
        
        text = response.choices[0].message.content.strip()
        
        # 提取 Python 代码
        code = ""
        if "```python" in text:
            start = text.index("```python") + 10
            end = text.index("```", start)
            code = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            code = text[start:end].strip()
        else:
            code = text
        
        if not code:
            raise HTTPException(status_code=400, detail="AI 未生成有效代码")
        
        # 在沙箱中执行代码（带异常保护）
        import traceback
        try:
            # ★ 注意：必须用同一个 dict 作为 globals 和 locals，
            #    否则 exec 内部对 df 的赋值不会写回 local_vars
            local_vars = {"df": df, "pd": pd, "np": np}
            exec(code, local_vars)
            df = local_vars['df']
        except Exception as exec_err:
            err_detail = traceback.format_exc()[-300:]
            raise HTTPException(status_code=400, detail=f"代码执行出错: {str(exec_err)}。可能列名与你提供的不匹配，请用更简单的描述重试。\n{err_detail}")
        
        # 获取新列
        new_columns = [c for c in df.columns if c not in original_columns]
        if not new_columns:
            raise HTTPException(status_code=400, detail="AI 计算后未生成新列，请尝试更具体的描述")
        
        # 更新 session
        manager.update_data(req.session_id, df)
        
        # 返回结果
        preview = df.head(10).replace({np.nan: None}).to_dict(orient="records")
        return sanitize_json({
            "success": True,
            "new_columns": new_columns,
            "message": f"已新增 {len(new_columns)} 个计算列：{'、'.join(new_columns)}",
            "preview": preview,
        })
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=enhance_ai_error(e, model=req.model or "", base_url=req.base_url or ""))


# ===== 同环比专用计算 =====

class TongHuanBiRequest(BaseModel):
    session_id: str
    value_column: str  # 数值列，如"销售金额"
    date_column: str = "日期"  # 日期列




def _safe_to_string(df_preview):
    """Safe DataFrame to string, avoiding GBK encoding issues on Windows."""
    import io
    buf = io.StringIO()
    df_preview.to_csv(buf, index=False, encoding='utf-8')
    return buf.getvalue()

@router.post("/data/tonghuanbi")
async def data_tonghuanbi(req: TongHuanBiRequest):
    """
    同环比专用计算接口：
    1. 按月聚合（sum）数值列
    2. 识别"本年"和"上年"（自动选取有数据的最近两个年份）
    3. 计算 同比增长率 = (本年值 - 上年同期值) / 上年同期值
    4. 计算 环比增长率 = (本期值 - 上期值) / 上期值
    5. 返回结构化表格数据
    """
    df = manager.get_data(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="未找到数据，请先上传文件")

    if req.value_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"数值列「{req.value_column}」不存在")

    if req.date_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"日期列「{req.date_column}」不存在")

    # 1. 复制需要的列，解析年月
    work = df[[req.date_column, req.value_column]].copy()
    work[req.date_column] = pd.to_datetime(work[req.date_column], errors='coerce')
    work = work.dropna(subset=[req.date_column])
    work['year'] = work[req.date_column].dt.year
    work['month'] = work[req.date_column].dt.month
    work['year_month'] = work[req.date_column].dt.strftime('%Y-%m')

    # 2. 按月聚合（sum）
    monthly = work.groupby(['year', 'month', 'year_month'])[req.value_column].sum().reset_index()
    monthly = monthly.sort_values(['year', 'month']).reset_index(drop=True)

    # 3. 找出有数据的年份
    years = sorted(monthly['year'].unique())
    if len(years) < 2:
        # 只有一年数据，只能做环比
        current_year = years[-1]
        months_single = monthly[monthly['year'] == current_year].copy()
        months_single['period_label'] = months_single['year_month']
        months_single['value'] = months_single[req.value_column]

        result_rows = []
        for _, row in months_single.iterrows():
            result_rows.append({
                'period': str(row['period_label']),
                '上年值': None,
                '本年值': round(float(row['value']), 2),
            })

        # 环比
        for i, row in enumerate(result_rows):
            if i == 0:
                row['环比增长率'] = None
            else:
                prev = result_rows[i - 1]['本年值']
                curr = row['本年值']
                if prev and prev != 0:
                    row['环比增长率'] = round((curr - prev) / prev, 4)
                else:
                    row['环比增长率'] = None
        # 没有同比
        chart_option = _build_tonghuanbi_line_chart(
            result_rows, req.value_column, str(current_year), None, has_yoy=False
        )
        return sanitize_json({
            "success": True,
            "value_column": req.value_column,
            "current_year": str(current_year),
            "previous_year": None,
            "rows": result_rows,
            "has_yoy": False,
            "chart_option": chart_option,
        })

    # 4. 选取最近两年（假设数据包含连续年份）
    current_year = years[-1]
    previous_year = years[-2]

    # 检查是否是真正的连续年份
    if current_year - previous_year > 1:
        # 年份不连续，降级为只看当前年
        return await data_tonghuanbi_single_year(monthly, current_year, req.value_column)

    # 构建 12 个月份的数据
    all_months = list(range(1, 13))
    prev_data = monthly[monthly['year'] == previous_year].set_index('month')[req.value_column].to_dict()
    curr_data = monthly[monthly['year'] == current_year].set_index('month')[req.value_column].to_dict()

    result_rows = []
    for m in all_months:
        prev_val = round(float(prev_data.get(m, 0)), 2) if m in prev_data else None
        curr_val = round(float(curr_data.get(m, 0)), 2) if m in curr_data else None

        # 跳过两个值都没有的月份
        if prev_val is None and curr_val is None:
            continue

        result_rows.append({
            'month': m,
            'period': f'{current_year}-{m:02d}',
            '上年值': prev_val,
            '本年值': curr_val,
        })

    # 5. 计算同比增长率
    for row in result_rows:
        prev = row['上年值']
        curr = row['本年值']
        if prev is not None and curr is not None and prev != 0:
            row['同比增长率'] = round((curr - prev) / prev, 4)
        else:
            row['同比增长率'] = None

    # 6. 计算环比增长率（本年内部 month-to-month）
    for i, row in enumerate(result_rows):
        if i == 0:
            row['环比增长率'] = None  # 第一行没有环比
        else:
            curr_val = row['本年值']
            prev_val = result_rows[i - 1]['本年值']
            if curr_val is not None and prev_val is not None and prev_val != 0:
                row['环比增长率'] = round((curr_val - prev_val) / prev_val, 4)
            else:
                row['环比增长率'] = None

    # 7. 生成同环比折线图 ECharts option
    chart_option = _build_tonghuanbi_line_chart(
        result_rows, req.value_column, str(current_year), str(previous_year), has_yoy=True
    )

    return sanitize_json({
        "success": True,
        "value_column": req.value_column,
        "current_year": str(current_year),
        "previous_year": str(previous_year),
        "rows": result_rows,
        "has_yoy": True,
        "chart_option": chart_option,
    })


async def data_tonghuanbi_single_year(monthly: pd.DataFrame, current_year: int, value_column: str):
    """辅助函数：只有单年数据时只计算环比"""
    months = monthly[monthly['year'] == current_year].sort_values('month')
    result_rows = []
    for _, row in months.iterrows():
        result_rows.append({
            'month': int(row['month']),
            'period': str(row['year_month']),
            '上年值': None,
            '本年值': round(float(row[value_column]), 2),
            '同比增长率': None,
        })
    for i, row in enumerate(result_rows):
        if i == 0:
            row['环比增长率'] = None
        else:
            curr = row['本年值']
            prev = result_rows[i - 1]['本年值']
            if prev and prev != 0:
                row['环比增长率'] = round((curr - prev) / prev, 4)
            else:
                row['环比增长率'] = None
    chart_option = _build_tonghuanbi_line_chart(
        result_rows, value_column, str(current_year), None, has_yoy=False
    )
    return sanitize_json({
        "success": True,
        "value_column": value_column,
        "current_year": str(current_year),
        "previous_year": None,
        "rows": result_rows,
        "has_yoy": False,
        "chart_option": chart_option,
    })


def _build_tonghuanbi_line_chart(
    rows: list,
    value_column: str,
    current_year: str,
    previous_year: str | None,
    has_yoy: bool,
) -> dict:
    """根据同环比表格数据，生成 ECharts 折线图 option"""
    months = [str(r.get('month', '')) + '月' if r.get('month') else r.get('period', '') for r in rows]

    series = []

    if has_yoy and previous_year:
        # 上年值折线
        prev_vals = [r.get('上年值', None) for r in rows]
        series.append({
            "name": f"{previous_year}年{value_column}",
            "type": "line",
            "data": prev_vals,
            "smooth": True,
            "lineStyle": {"width": 2, "type": "dashed"},
            "itemStyle": {"color": "#94a3b8"},
            "symbol": "circle",
            "symbolSize": 6,
        })

    # 本年值折线（主趋势线）
    curr_vals = [r.get('本年值', None) for r in rows]
    series.append({
        "name": f"{current_year}年{value_column}",
        "type": "line",
        "data": curr_vals,
        "smooth": True,
        "lineStyle": {"width": 3},
        "itemStyle": {"color": "#22d3ee"},
        "areaStyle": {
            "color": {
                "type": "linear",
                "x": 0, "y": 0, "x2": 0, "y2": 1,
                "colorStops": [
                    {"offset": 0, "color": "rgba(34,211,238,0.25)"},
                    {"offset": 1, "color": "rgba(34,211,238,0.02)"},
                ],
            }
        },
        "symbol": "circle",
        "symbolSize": 8,
    })

    title_text = f"{current_year}年 vs {previous_year}年 {value_column}月度趋势" if has_yoy and previous_year else f"{current_year}年 {value_column}月度趋势"

    option = {
        "title": {
            "text": title_text,
            "textStyle": {"color": "#f8fafc", "fontSize": 14, "fontWeight": "bold"},
            "left": "center",
        },
        "tooltip": {
            "trigger": "axis",
            "backgroundColor": "rgba(15,23,42,0.92)",
            "borderColor": "#334155",
            "textStyle": {"color": "#f8fafc", "fontSize": 12},
        },
        "legend": {
            "data": [s["name"] for s in series],
            "bottom": 0,
            "textStyle": {"color": "#94a3b8", "fontSize": 11},
        },
        "grid": {"left": "8%", "right": "6%", "top": "15%", "bottom": "12%"},
        "xAxis": {
            "type": "category",
            "data": months,
            "axisLabel": {"color": "#94a3b8", "fontSize": 10},
            "axisLine": {"lineStyle": {"color": "#334155"}},
            "axisTick": {"show": False},
        },
        "yAxis": {
            "type": "value",
            "name": value_column,
            "nameTextStyle": {"color": "#94a3b8", "fontSize": 10},
            "axisLabel": {
                "color": "#94a3b8",
                "fontSize": 10,
                "formatter": "{value}",
            },
            "splitLine": {"lineStyle": {"color": "rgba(148,163,184,0.1)"}},
        },
        "series": series,
    }

    # 格式化 Y 轴（万/亿）
    max_val = max((v for r in rows for v in [r.get('上年值'), r.get('本年值')] if v), default=0)
    if max_val >= 100_000_000:
        option["yAxis"]["axisLabel"]["formatter"] = "function(v) { return (v / 100000000).toFixed(1) + '亿'; }"
    elif max_val >= 10_000:
        option["yAxis"]["axisLabel"]["formatter"] = "function(v) { return (v / 10000).toFixed(1) + '万'; }"

    return option


# ===== 数据导出 =====

class DownloadRequest(BaseModel):
    session_id: str
    export_original: bool = False  # True=导出原始数据, False=导出当前数据




def _safe_to_string(df_preview):
    """Safe DataFrame to string, avoiding GBK encoding issues on Windows."""
    import io
    buf = io.StringIO()
    df_preview.to_csv(buf, index=False, encoding='utf-8')
    return buf.getvalue()

@router.post("/data/download")
async def data_download(req: DownloadRequest):
    """导出当前数据（或原始数据）为 CSV"""
    if req.export_original:
        df = manager.get_original_data(req.session_id)
        if df is None:
            if manager.get_session(req.session_id) is None:
                raise HTTPException(status_code=404, detail="未找到数据，请先上传文件")
            raise HTTPException(status_code=404, detail="原始数据已释放，请重新上传")
    else:
        df = manager.get_data(req.session_id)
        if df is None:
            raise HTTPException(status_code=404, detail="未找到数据，请先上传文件")

    # 生成 CSV
    stream = io.StringIO()
    df.to_csv(stream, index=False, encoding='utf-8-sig')
    content = stream.getvalue()
    stream.close()

    prefix = "原始" if req.export_original else "清洗后"
    raw_name = f"{prefix}数据_{req.session_id[:8]}.csv"
    # Starlette 用 latin-1 编码 header，中文 filename 会触发 UnicodeEncodeError(500)。
    # 按 RFC 5987：filename 退化 ASCII 名，中文名走 filename*=UTF-8''。
    ascii_name = f"{'original' if req.export_original else 'cleaned'}_data_{req.session_id[:8]}.csv"
    disposition = f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(raw_name)}"

    return StreamingResponse(
        io.BytesIO(content.encode('utf-8-sig')),
        media_type="text/csv",
        headers={"Content-Disposition": disposition},
    )


@router.get("/data/download")
async def data_download_get(session_id: str, export_original: bool = False):
    """导出（GET 入口，供前端 <a> 原生下载，规避 blob.click 在部分浏览器不触发的坑）。逻辑复用 POST 版本。"""
    return await data_download(DownloadRequest(session_id=session_id, export_original=export_original))


class DatasetSelectRequest(BaseModel):
    session_id: str
    dataset_id: str


class DatasetRemoveRequest(BaseModel):
    session_id: str
    dataset_id: str


class ApiConfigRequest(BaseModel):
    session_id: str
    api_key: str = ""
    ai_provider: str = ""
    custom_model: str = ""
    custom_base_url: str = ""


@router.post("/data/select")
async def data_select(req: DatasetSelectRequest):
    """切换当前分析对象（active 数据集）；按需 reload 内存 df + 释放其余非 active 内存 df"""
    ok = manager.select_dataset(req.session_id, req.dataset_id)
    if not ok:
        logger.warning("[select_dataset] failed sid=%s did=%s", req.session_id, req.dataset_id)
        raise HTTPException(status_code=404, detail="数据集不存在")
    logger.info("[select_dataset] ok sid=%s did=%s", req.session_id, req.dataset_id)
    return sanitize_json({"success": True, "active_dataset_id": req.dataset_id})


@router.get("/data/datasets")
async def data_datasets(session_id: str):
    """返回会话全部数据集元信息（供前端"已上传报表"列表 / 刷新拉回）+ 累计额度"""
    try:
        session = manager.get_session(session_id)
        used = session.uploaded_bytes if session else 0
        # API 配置整 session 一份，放 response 根级（与 used_bytes 同级），随刷新一并拉回
        api_cfg = manager.get_api_config(session_id)
        # 数据集数量配额：登录用户走 users.dataset_limit；未登录/游客不限制（返 None）
        datasets_list = manager.get_datasets(session_id)
        dataset_limit = None
        try:
            from backend.db import crud as _crud
            owner = session.user_id if session else None
            if owner:
                dataset_limit = _crud.get_user_dataset_limit(int(owner))
        except Exception:
            pass
        return sanitize_json({
            "success": True,
            "datasets": datasets_list,
            "used_bytes": used,
            "quota_bytes": QUOTA_BYTES,
            "dataset_count": len(datasets_list),
            "dataset_limit": dataset_limit,
            "api_key": api_cfg["api_key"],
            "ai_provider": api_cfg["ai_provider"],
            "custom_model": api_cfg["custom_model"],
            "custom_base_url": api_cfg["custom_base_url"],
        })
    except Exception as e:
        # 抓全量堆栈到服务端日志，并以可读 JSON 返回，避免静默崩成 500(ERR_BAD_RESPONSE)
        logger.error("[data/datasets] 列出数据集失败 session_id=%s: %s\n%s",
                     session_id, e, traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content=sanitize_json({"success": False, "error": f"列出数据集失败: {e}"}),
        )


@router.post("/data/remove-dataset")
async def data_remove_dataset(req: DatasetRemoveRequest):
    """删除指定数据集（删落盘 + 减额度 + 回退 active）；修复五"""
    ok = manager.remove_dataset(req.session_id, req.dataset_id)
    if not ok:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return sanitize_json({"success": True})


@router.post("/data/api-config")
async def data_save_api_config(req: ApiConfigRequest):
    """保存整套 AI 配置（api_key/ai_provider/custom_model/custom_base_url）进 session 并落库"""
    try:
        manager.set_api_config(
            req.session_id,
            req.api_key,
            req.ai_provider,
            req.custom_model,
            req.custom_base_url,
        )
        return sanitize_json({"success": True})
    except Exception as e:
        logger.error("[data/api-config] 保存 AI 配置失败 session_id=%s: %s\n%s",
                     req.session_id, e, traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content=sanitize_json({"success": False, "error": f"保存 AI 配置失败: {e}"}),
        )
