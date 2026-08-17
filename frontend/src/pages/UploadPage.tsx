import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, Trash2, AlertTriangle, Database, GitMerge, ArrowRight } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import DataTable from '../components/DataTable';
import { useData } from '../contexts/DataContext';
import { formatBytes } from '../utils/format';
import {
  uploadFile, listDatasets, removeDataset, selectDataset, clearData,
} from '../api/client';
import type { DatasetInfo } from '../types/api';

const QUOTA_DEFAULT = 30 * 1024 * 1024;
const ACCEPT = '.csv, .xlsx, .xls, .json, .sqlite, .db';

export default function UploadPage() {
  const { state, dispatch, ensureValidSession } = useData();
  const navigate = useNavigate();
  const { sessionId, datasets, activeDatasetId, usedBytes, quotaBytes, datasetCount, datasetLimit, fileName, rows, columns, preview, columnInfo } = state;
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await listDatasets(sessionId);
        if (!alive) return;
        dispatch({ type: 'SET_DATASETS', datasets: res.datasets });
        dispatch({
          type: 'SET_QUOTA',
          usedBytes: res.used_bytes,
          quotaBytes: res.quota_bytes,
          datasetCount: res.dataset_count,
          datasetLimit: res.dataset_limit ?? null,
        });
        const active = res.datasets.find((d) => d.is_active);
        if (active) dispatch({ type: 'SELECT_DATASET', datasetId: active.dataset_id });
      } catch { /* 会话暂无数据，忽略 */ }
    })();
    return () => { alive = false; };
  }, [sessionId]);

  const doUpload = useCallback(async (file: File) => {
    if (state.usedBytes + file.size > (state.quotaBytes || QUOTA_DEFAULT)) {
      throw new Error('累计上传额度已满，无法继续上传');
    }
    setUploadError(null);
    try {
      const res = await uploadFile(file, sessionId);
      const items = (res.datasets && res.datasets.length)
        ? res.datasets
        : [{
            dataset_id: res.dataset_id,
            file_name: res.file_name ?? file.name,
            rows: res.rows,
            columns: res.columns,
            memory_usage: res.memory_usage,
            total_missing: res.total_missing,
            duplicate_rows: res.duplicate_rows,
            preview: res.preview,
            column_info: res.column_info,
            column_names: (res.column_info?.map((c) => c.name)) ?? [],
          }];
      items.forEach((d, i) => {
        const ds: DatasetInfo = {
          dataset_id: d.dataset_id,
          file_name: d.file_name ?? file.name,
          file_size_bytes: i === 0 ? res.file_size_bytes : 0,
          rows: d.rows,
          columns: d.column_names ?? [],
          column_info: d.column_info,
          preview: d.preview,
          uploaded_at: Date.now(),
          is_active: i === 0,
        };
        dispatch({ type: 'ADD_DATASET', payload: ds });
      });
      dispatch({
        type: 'SET_QUOTA',
        usedBytes: res.used_bytes,
        quotaBytes: res.quota_bytes,
        datasetCount: res.dataset_count,
        datasetLimit: res.dataset_limit ?? null,
      });
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || '上传失败';
      throw new Error(msg);
    }
  }, [sessionId, state.usedBytes, state.quotaBytes]);

  const handleSelect = useCallback(async (datasetId: string) => {
    try { await selectDataset(sessionId, datasetId); } catch { /* ignore */ }
    dispatch({ type: 'SELECT_DATASET', datasetId });
  }, [sessionId]);

  const handleRemove = useCallback(async (datasetId: string) => {
    try { await removeDataset(sessionId, datasetId); } catch { /* ignore */ }
    try {
      const res = await listDatasets(sessionId);
      dispatch({ type: 'SET_DATASETS', datasets: res.datasets });
      dispatch({
        type: 'SET_QUOTA',
        usedBytes: res.used_bytes,
        quotaBytes: res.quota_bytes,
        datasetCount: res.dataset_count,
        datasetLimit: res.dataset_limit ?? null,
      });
      const active = res.datasets.find((d) => d.is_active);
      if (active) dispatch({ type: 'SELECT_DATASET', datasetId: active.dataset_id });
    } catch { /* ignore */ }
  }, [sessionId]);

  const handleRelease = async () => {
    if (!confirm('确定结束会话？该会话的全部数据（上传数据、清洗结果、分析产物、已保存图表）将被彻底清空。')) return;
    await clearData(sessionId);
    dispatch({ type: 'CLEAR_DATA' });
    dispatch({ type: 'SET_QUOTA', usedBytes: 0, quotaBytes: 0, datasetCount: 0, datasetLimit: null });
    // 结束后自动创建新会话，避免死守已清空的旧 id
    const newId = await ensureValidSession();
    alert(`会话已结束，已创建新会话 ${newId}`);
  };

  const handleDeleteAll = useCallback(async () => {
    if (!confirm('确定删除全部已上传报表？此操作不可撤销。')) return;
    setDeleting(true);
    try {
      for (const ds of datasets) {
        await removeDataset(sessionId, ds.dataset_id);
      }
      const res = await listDatasets(sessionId);
      dispatch({ type: 'SET_DATASETS', datasets: res.datasets });
      dispatch({
        type: 'SET_QUOTA',
        usedBytes: res.used_bytes,
        quotaBytes: res.quota_bytes,
        datasetCount: res.dataset_count,
        datasetLimit: res.dataset_limit ?? null,
      });
      const active = res.datasets.find((d) => d.is_active);
      if (active) dispatch({ type: 'SELECT_DATASET', datasetId: active.dataset_id });
    } catch { /* ignore */ }
    finally { setDeleting(false); }
  }, [sessionId, datasets]);

  // 拖拽上传（原生实现，无额外依赖）
  const onDrop = async (files: FileList | null) => {
    setDragging(false);
    if (!files || files.length === 0) return;
    for (const file of Array.from(files)) {
      try {
        await doUpload(file);
      } catch (e: any) {
        setUploadError(e?.message || '上传失败');
      }
    }
  };

  const quota = quotaBytes || QUOTA_DEFAULT;
  const pct = quota > 0 ? Math.min(100, (usedBytes / quota) * 100) : 0;
  const full = usedBytes >= quota;
  const warn = pct > 80;
  const barColor = full ? '#fb7185' : warn ? '#fbbf24' : '#8b5cf6';

  return (
    <div className="page-enter pt-10">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-white/70 border border-violet-200 text-violet-600 shadow-[0_4px_14px_rgba(139,92,246,0.18)]">
          <Database className="w-5 h-5" />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Data Uploads</h1>
          <p className="text-sm text-slate-500 mt-0.5">拖拽或浏览上传你的数据文件</p>
        </div>
      </div>

      {/* 额度进度条（全透明） */}
      <div className="p-4 mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-slate-500">上传额度</span>
          <span className="text-sm font-medium" style={{ color: full ? '#e11d48' : warn ? '#d97706' : '#7c3aed' }}>
            已用 {formatBytes(usedBytes)} / {formatBytes(quota)}
          </span>
        </div>
        <div className="h-2 rounded-full bg-slate-200/70 overflow-hidden">
          <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, background: barColor }} />
        </div>
        {/* 数据集数量配额（仅登录用户有 datasetLimit） */}
        {datasetLimit !== null && datasetLimit !== undefined && (
          <div className="mt-3 flex items-center gap-2 text-xs">
            <Database size={12} className="text-violet-500" />
            <span className="text-slate-600">数据集</span>
            <span
              className="font-medium"
              style={{ color: datasetCount >= datasetLimit ? '#e11d48' : '#7c3aed' }}
            >
              {datasetCount} / {datasetLimit}
            </span>
            {datasetCount >= datasetLimit && (
              <span className="text-rose-500">已满，请在下方列表删除部分数据集后重试</span>
            )}
          </div>
        )}
        {full && <p className="text-xs text-rose-500 mt-2">额度已用尽，请删除部分报表或释放插槽。</p>}
      </div>

      {/* 环形上传区（透明玻璃圈，对齐空壳） */}
      <div className="flex flex-col items-center justify-center py-4">
        <label
          className={`donut-drop ${dragging ? 'drag-active' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); onDrop(e.dataTransfer.files); }}
        >
          <div className="donut-mask-wrapper">
            <div className="frosted-donut" />
          </div>
          <div className="outer-border" />
          <div className="inner-border">
            <div className="relative z-[3] flex flex-col items-center text-center px-8 pointer-events-none">
              <svg className="doc-icon" viewBox="0 0 24 24" width="54" height="54">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
              <p className="mt-4 text-slate-700 font-medium">拖拽文件到此处 或 点击浏览</p>
              <p className="mt-1 text-xs text-slate-600">支持 CSV · Excel · JSON · SQLite</p>
            </div>
          </div>
          <input
            type="file"
            accept={ACCEPT}
            multiple
            className="hidden"
            disabled={full}
            onChange={(e) => onDrop(e.target.files)}
          />
        </label>
        {uploadError && (
          <div className="mt-4 glass-card px-4 py-2.5 text-sm text-rose-500 flex items-center gap-2">
            <AlertTriangle size={14} />{uploadError}
          </div>
        )}
      </div>

      {/* 已上传报表列表 */}
      <div className="glass-card-soft p-5 mt-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold flex items-center gap-2 text-slate-900">
            <Database size={18} className="text-violet-500" />已上传报表
            <span className="text-xs text-slate-600">（{datasets.length}）</span>
          </h2>
          <button
            onClick={handleDeleteAll}
            disabled={deleting || datasets.length === 0}
            className="px-4 py-2 rounded-lg text-sm font-medium text-white bg-violet-600 hover:bg-violet-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {deleting ? '删除中…' : '一键删除'}
          </button>
        </div>

        {datasets.length === 0 ? (
          <p className="text-sm text-slate-600 py-6 text-center">暂无报表，请拖拽文件上传。</p>
        ) : (
          <div className="flex flex-col gap-3">
            {datasets.map((ds) => {
              const isActive = ds.dataset_id === activeDatasetId;
              return (
                <div
                  key={ds.dataset_id}
                  onClick={() => handleSelect(ds.dataset_id)}
                  className={`relative flex items-center justify-between p-4 rounded-xl cursor-pointer transition-all border ${
                    isActive ? 'border-violet-300 bg-violet-50/70' : 'border-slate-200 bg-white/50 hover:border-violet-300'
                  }`}
                >
                  {isActive && <div className="absolute left-0 top-0 bottom-0 w-1 rounded-l-xl bg-violet-500" />}
                  <div className="flex items-center gap-3 min-w-0">
                    <FileText size={20} className="text-violet-500 shrink-0" />
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="font-medium text-slate-800 truncate">{ds.file_name}</p>
                        {isActive ? (
                          <span className="text-[11px] px-2 py-0.5 rounded-full bg-violet-100 text-violet-700 border border-violet-200 font-medium">Active</span>
                        ) : (
                          <span className="text-[11px] px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 border border-slate-200">点击选用</span>
                        )}
                      </div>
                      {ds.is_merged && (
                        <span className="mt-1 inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full bg-violet-100 text-violet-700 border border-violet-200">
                          <GitMerge size={11} />合并宽表
                          {ds.merge_keys && ds.merge_keys.length > 0 ? ` · 按 ${ds.merge_keys.join('/')} 关联` : ''}
                          {ds.sources && ds.sources.length > 0 ? ` · 来源${ds.sources.length}表` : ''}
                        </span>
                      )}
                      <p className="text-xs text-slate-500 mt-0.5">{formatBytes(ds.file_size_bytes)} · {ds.rows} 行 × {ds.columns.length} 列</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 shrink-0" onClick={(e) => e.stopPropagation()}>
                    <button
                      onClick={() => handleRemove(ds.dataset_id)}
                      className="p-1.5 rounded-lg text-slate-600 hover:text-rose-500 hover:bg-rose-50 transition-colors"
                      title="删除该报表"
                    ><Trash2 size={16} /></button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* 当前数据集概览与预览 */}
      {fileName ? (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-slate-900">数据概览</h2>
            <button
              onClick={() => navigate('/analysis')}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-semibold text-white bg-violet-600 hover:bg-violet-700 transition-colors"
            >
              进入分析 <ArrowRight className="w-4 h-4" />
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <MetricCard label="当前报表" value={fileName} hint="文件名" className="glass-card-soft" />
            <MetricCard label="总行数" value={rows.toString()} hint="数据规模" className="glass-card-soft" />
            <MetricCard label="总列数" value={columns.toString()} hint="字段数量" className="glass-card-soft" />
          </div>
          {columnInfo && columnInfo.length > 0 && (
            <div className="glass-card-soft p-5 mb-6">
              <h3 className="text-base font-medium mb-4 text-slate-800">字段信息</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-slate-500 border-b border-slate-200">
                      <th className="py-2 pr-4">字段名</th><th className="py-2 pr-4">类型</th>
                      <th className="py-2 pr-4">缺失值</th><th className="py-2 pr-4">唯一值</th><th className="py-2">示例</th>
                    </tr>
                  </thead>
                  <tbody>
                    {columnInfo.map((c, i) => (
                      <tr key={i} className="border-b border-slate-100">
                        <td className="py-2 pr-4 text-slate-800">{c.name}</td>
                        <td className="py-2 pr-4 text-slate-500">{c.dtype}</td>
                        <td className="py-2 pr-4 text-slate-500">{c.missing}（{c.missing_rate}%）</td>
                        <td className="py-2 pr-4 text-slate-500">{c.unique}</td>
                        <td className="py-2 text-slate-500 truncate max-w-xs">{c.sample}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          <div className="glass-card-soft p-5">
            <h3 className="text-base font-medium mb-4 text-slate-800">数据预览（前 100 行）</h3>
            <DataTable data={preview} maxRows={100} />
          </div>
        </div>
      ) : (
        <p className="text-sm text-slate-600 mt-6 text-center py-6">请选择一张报表以查看预览。</p>
      )}

      {/* 底部操作栏 */}
      <div className="mt-6 flex justify-end">
        <button
          onClick={handleRelease}
          className="px-4 py-2 rounded-lg text-sm font-medium bg-rose-500/10 text-rose-600 hover:bg-rose-500/20 transition-colors"
        >结束会话 / 释放插槽</button>
      </div>
    </div>
  );
}
