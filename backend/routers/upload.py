"""
文件上传 API 路由
"""
import io
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body, Header
from pydantic import BaseModel
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.data_loader import load_csv, load_json, load_sqlite, get_data_info, get_column_info, identify_excel_data_sheets
from backend.services.session_manager import manager
from backend.services.auth import get_optional_user
from config import MAX_FILE_SIZE_BYTES, MAX_UPLOAD_SIZE_MB, QUOTA_BYTES

router = APIRouter()

# P0（内存画像结论三）：限制并发文件解析，避免 XLSX 解析瞬时 RSS 尖峰（×3.9）叠加导致 OOM。
# 解析从事件循环移入线程池（asyncio.to_thread），既释放事件循环避免大文件卡住其他请求，
# 又通过信号量把「同时解析」限制为小并发，使 xlsx 尖峰重叠 ≤ 3×33MB，远低于 350MB 可用池。
# 仅对尖峰风险高的 xlsx / sqlite 限流；csv / json 膨胀低（×1.1）直接线程池解析，不占信号量。
_PARSE_SEMAPHORE = asyncio.Semaphore(3)


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


class UploadResponse(BaseModel):
    session_id: str
    success: bool
    file_name: str
    rows: int
    columns: int
    preview: list
    column_info: list
    dataset_id: str = ""
    used_bytes: int = 0
    quota_bytes: int = 0
    file_size_bytes: int = 0


@router.post("/upload/gate")
async def upload_gate(session_id: str = Body(..., embed=True)):
    """预约数据插槽闸门：在真正传文件前调用。

    - 有空位：返回 {"granted": true, "session_id"}，前端直接上传。
    - 满员：返回 {"granted": false, "ticket_id", "position"}，前端进入排队弹窗并轮询。
    统一用 200 + 结构化 JSON，避免触发响应拦截器对 429 结构的破坏。
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    return manager.acquire_for_upload(session_id)


@router.get("/upload/queue/{ticket_id}")
async def queue_status(ticket_id: str):
    """查询排队状态：ready（附 session_id）/ queued（附 position）/ expired。"""
    return manager.queue_status(ticket_id)


@router.post("/upload/queue/cancel")
async def cancel_queue(ticket_id: str = Body(..., embed=True)):
    """取消排队：尽力从等待队列移除票据。"""
    if not ticket_id:
        raise HTTPException(status_code=400, detail="缺少 ticket_id")
    manager.cancel_queue(ticket_id)
    return {"success": True}


@router.post("/upload/release")
async def release_slot(session_id: str = Body(..., embed=True)):
    """释放某会话的数据插槽（丢弃 DataFrame/原文件以释放内存），并自动晋升队首。

    这是「自动入队」的关键触发点：某用户结束使用、离开或清空数据后调用，
    排队中的用户即可获得空位并开始上传。
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    released = manager.release_slot(session_id)
    return {"success": True, "released": released}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form(""),
                     authorization: Optional[str] = Header(default=None)):
    """
    上传数据文件
    支持 CSV/Excel/JSON/SQLite 格式
    返回数据预览和字段信息；每张文件作为独立数据集追加（不覆盖旧表）
    """
    # 解析登录态（游客可为 None；登录用户把数据归属到其 user_id，B1 修复）
    current_user = get_optional_user(authorization)
    current_user_id = str(current_user["id"]) if current_user else None

    # 创建或使用已有会话（前置，便于累计额度判断）
    if not session_id:
        # 正常路径前端必带 sessionId（已预占插槽），此兜底理论不触发；仍计入上限以保一致
        session_id = manager.create_session()
        manager.reserve_session(session_id)

    # 验证文件大小（单文件上限）
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"文件大小超过 {MAX_UPLOAD_SIZE_MB}MB 限制")

    # 累计 30MB 额度拦截（所有已上传文件字节之和）
    session = manager.get_session(session_id)
    used = session.uploaded_bytes if session else 0
    if used + len(content) > QUOTA_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"累计上传额度已用尽：已用 {used / 1024 / 1024:.1f}MB / 剩余 {(QUOTA_BYTES - used) / 1024 / 1024:.1f}MB。请删除部分报表或释放插槽。",
        )

    # 验证文件格式
    filename = file.filename.lower()
    ext = filename.rsplit('.', 1)[-1] if '.' in filename else ''
    supported = {'csv', 'xlsx', 'xls', 'json', 'db', 'sqlite'}
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: .{ext}。支持: CSV, Excel, JSON, SQLite"
        )

    try:
        # 加载数据（解析移入线程池 + 信号量限流，见 _PARSE_SEMAPHORE 说明）
        # 解析为若干 (sheet_name, df) 候选；Excel 自动识别全部数据表 sheet
        if ext == 'csv':
            sheets = [(None, await asyncio.to_thread(load_csv, content, file.filename))]
        elif ext in ('xlsx', 'xls'):
            async with _PARSE_SEMAPHORE:
                identified = await asyncio.to_thread(identify_excel_data_sheets, content)
            if not identified:
                raise HTTPException(
                    status_code=400,
                    detail="Excel 中未识别到任何数据表（可能都是空表或说明页）",
                )
            sheets = [(s["sheet_name"], s["df"]) for s in identified]
        elif ext == 'json':
            sheets = [(None, await asyncio.to_thread(load_json, content))]
        elif ext in ('db', 'sqlite'):
            async with _PARSE_SEMAPHORE:
                tables = await asyncio.to_thread(load_sqlite, content)
            # 取第一个表作为数据
            if isinstance(tables, dict):
                first_table = list(tables.keys())[0]
                sheets = [(first_table, tables[first_table])]
            else:
                sheets = [(None, tables)]
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式")

        # 逐个数据表落库（多 sheet Excel → 多个独立数据集）
        created = []          # [{dataset_id, file_name, rows, columns, ...}] 供前端批量建表
        first_meta = None
        for idx, (sheet_name, df) in enumerate(sheets):
            if df is None or df.empty:
                continue
            preview = df.head(100).replace({np.nan: None}).to_dict(orient="records")
            column_info = get_column_info(df)
            data_info = get_data_info(df)

            # 转换列信息为列表
            columns_list = []
            for _, row in column_info.iterrows():
                columns_list.append({
                    "name": str(row.get("列名", row.get("column", ""))),
                    "dtype": str(row.get("数据类型", row.get("dtype", ""))),
                    "missing": int(row.get("缺失值", row.get("missing", 0))),
                    "missing_rate": _parse_missing_rate(row),
                    "unique": int(row.get("唯一值数", row.get("unique", 0))),
                    "sample": str(row.get("示例值", row.get("sample", ""))),
                })

            # 多 sheet：文件名带 sheet 标识；首个 sheet 计全额额度，其余只落库不计额度
            ds_name = file.filename if sheet_name is None else f"{file.filename} ▸ {sheet_name}"
            dataset_id = manager.add_dataset(
                session_id, df,
                file_name=ds_name,
                file_size_bytes=len(content) if idx == 0 else 0,
                rows=int(data_info.get("行数", len(df))),
                columns=list(df.columns),
                column_info=columns_list,
                preview=preview,
                set_active=(idx == 0),
                account_quota=(idx == 0),
                user_id=current_user_id,
            )
            meta = {
                "dataset_id": dataset_id,
                "file_name": ds_name,
                "rows": int(data_info.get("行数", len(df))),
                "columns": int(data_info.get("列数", len(df.columns))),
                "memory_usage": str(data_info.get("内存占用", data_info.get("memory_usage", ""))),
                "total_missing": int(data_info.get("缺失值总数", data_info.get("total_missing", 0))),
                "duplicate_rows": int(data_info.get("重复行数", data_info.get("duplicate_rows", 0))),
                "preview": preview,
                "column_info": columns_list,
                "column_names": list(df.columns),
            }
            created.append(meta)
            if first_meta is None:
                first_meta = meta

            # 上传即侦察：扫描 df 结构存 session，供 Chat 智能体后续直接用
            try:
                from src.data_recon import scan
                session = manager.get_session(session_id)
                if session is not None:
                    session.data_profile = scan(df)
            except Exception:
                pass

        if not created:
            raise HTTPException(status_code=400, detail="文件内容为空或无法读取")

        # 多表懒自动合并：本次上传后立刻探测 session 中所有非合并数据集，
        # 若 ≥2 张且尚无 merged 宽表，则静默生成「合并宽表」并注册进 session。
        # 这样用户上传两表后无需发消息，下拉里就能立即看到宽表选项（Q1=B）。
        # 失败/无关联键静默跳过，不影响上传响应。
        try:
            from backend.services.multi_table import maybe_auto_merge
            merge_result = maybe_auto_merge(manager, session_id)
            if merge_result.get("status") == "merged":
                import logging as _lg
                _lg.getLogger("uvicorn.error").info(
                    f"[upload] 多表自动合并成功: dataset_id={merge_result.get('dataset_id')} "
                    f"keys={merge_result.get('keys')} rows={merge_result.get('rows')}"
                )
        except Exception as e:
            import logging as _lg
            _lg.getLogger("uvicorn.error").warning(
                f"[upload] 多表自动合并探测被跳过: {type(e).__name__}: {e}"
            )

        used_after = manager.get_session(session_id).uploaded_bytes
        return {
            "session_id": session_id,
            "dataset_id": first_meta["dataset_id"],
            "success": True,
            "used_bytes": used_after,
            "quota_bytes": QUOTA_BYTES,
            "file_size_bytes": len(content),
            "file_name": first_meta["file_name"],
            "rows": first_meta["rows"],
            "columns": first_meta["columns"],
            "memory_usage": first_meta["memory_usage"],
            "total_missing": first_meta["total_missing"],
            "duplicate_rows": first_meta["duplicate_rows"],
            "preview": first_meta["preview"],
            "column_info": first_meta["column_info"],
            "column_names": first_meta["column_names"],
            # 多 sheet：返回所有被识别出的数据表清单（单表时长度为 1，前端统一走列表）
            "datasets": created,
            "sheet_count": len(created),
        }

    except ValueError as e:
        import traceback as _tb
        import logging as _lg
        _lg.getLogger("uvicorn.error").error(
            f"Upload ValueError: {e}\n{_tb.format_exc()}"
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 配额超额：crud.save_dataset 抛 QuotaExceededError → 403 QUOTA_EXCEEDED（建议 3）
        from backend.db.crud import QuotaExceededError
        if isinstance(e, QuotaExceededError):
            raise HTTPException(status_code=403, detail="QUOTA_EXCEEDED: " + str(e))
        import traceback as _traceback
        import logging as _logging
        tb = _traceback.format_exc()
        _logging.getLogger("uvicorn.error").error(f"Upload error: {type(e).__name__}: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")
