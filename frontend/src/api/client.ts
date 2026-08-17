/* DataMind AI - Axios API 客户端 */

import axios from 'axios';
import type {
  UploadResponse, PreviewResponse, StatsResponse,
  InsightsResponse, ChatResponse, ChatSendResponse,
  AIReportResponse, KPIResponse, EChartResponse, EChartItem,
  DatasetInfo, DatasetListResponse,
  ProcessSubmitResponse, ProcessStatusResponse,
  AICleanSubmitResponse, AICleanStatusResponse,
} from '../types/api';
import type { ChartConfig } from '../types';
import type { SmartLayoutResponse } from './../types/dashboard';

// 部署时通过环境变量指定后端地址，本地开发走 Vite proxy
let API_BASE = import.meta.env.VITE_API_BASE || '/api';
// 兼容只写根域名的情况：统一追加 /api，避免上传接口 405
API_BASE = API_BASE.replace(/\/$/, '');
if (!API_BASE.endsWith('/api')) {
  API_BASE += '/api';
}

const api = axios.create({
  baseURL: API_BASE,
  timeout: 300000,  // 5 分钟，AI 清洗/报告生成需要更长时间
  // 不设置全局 Content-Type，让 axios 自动处理：
  // JSON 请求会自动设为 application/json，FormData 上传会自动设为 multipart/form-data
});

// 统一将 AI 模型名转为小写，避免大小写不匹配导致 model_not_found
// （阿里云百炼 / DeepSeek / OpenAI 等 API 的模型 ID 均为小写格式，如 qwen3.7-plus、deepseek-chat）
api.interceptors.request.use((config) => {
  if (
    config.data &&
    typeof config.data === 'object' &&
    typeof config.data.model === 'string'
  ) {
    config.data.model = config.data.model.toLowerCase();
  }
  // 注入登录态：已登录用户的所有业务请求自动携带 Bearer token
  const token = localStorage.getItem('datamind_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// 401 统一处理：业务接口先尝试 refresh 续期并重试一次；失败（refresh 过期）才跳登录
import { tryRefresh } from '../lib/api';
api.interceptors.response.use(
  (res) => res,
  async (err) => {
    const status = err.response?.status;
    const url = err.config?.url || '';
    const isAuthCall = url.includes('/auth/');
    const isSessionCall = url.includes('/session/');
    if (status === 401 && !isAuthCall && !isSessionCall) {
      const newToken = await tryRefresh();
      if (newToken && err.config) {
        err.config.headers.Authorization = `Bearer ${newToken}`;
        return api.request(err.config);
      }
      // 续期失败：记录来源并跳登录
      const current = window.location.pathname + window.location.search + window.location.hash;
      sessionStorage.setItem('datamind_post_login_redirect', current);
      if (!window.location.pathname.startsWith('/login') &&
          !window.location.pathname.startsWith('/register')) {
        window.location.href = '/login';
      }
    }
    // 原有网络错误重试逻辑在下方拦截器中继续处理
    return Promise.reject(err);
  },
);

// 后端休眠/无响应时的自动重试（同时支持 Render 部署版和本地 dev）
let _wakePromise: Promise<void> | null = null;

async function wakeUpBackend(reason: string): Promise<void> {
  if (_wakePromise) return _wakePromise;
  _wakePromise = (async () => {
    console.warn(`🔧 检测到后端无响应（${reason}），等待 5 秒后重试...`);
    // 等待一段时间让后端恢复（本地 vite proxy / Render 冷启动都适用）
    await new Promise(r => setTimeout(r, 5000));
    console.log('✅ 等待结束，即将重试请求');
  })();
  await _wakePromise;
  _wakePromise = null;
}

api.interceptors.response.use(
  (res) => res,
  async (err) => {
    // 检测网络/超时错误
    const isNetworkError = !err.response &&
      (err.code === 'ECONNABORTED' || err.code === 'ERR_NETWORK' ||
       err.message?.includes('timeout') || err.message?.includes('Network Error'));

    if (isNetworkError && err.config && !err.config._retried) {
      err.config._retried = true;
      await wakeUpBackend(err.message || err.code || 'unknown');
      return api(err.config);  // 重试
    }

    let msg = err.response?.data?.detail || err.message || '请求失败';
    // FastAPI 422 验证错误时 detail 是数组，需要提取第一条信息
    if (Array.isArray(msg)) {
      msg = msg.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join('; ');
    }
    if (typeof msg !== 'string') msg = String(msg);
    // 详细诊断日志：url + status + code
    const url = err.config?.url || '?';
    const status = err.response?.status ?? 'N/A';
    const code = err.code || 'N/A';
    console.error(`[API Error] ${url} → ${status} (${code}):`, msg);
    return Promise.reject(new Error(msg));
  }
);

/* ===== 会话 ===== */
export const createSession = async (): Promise<string> => {
  const { data } = await api.get('/session/new');
  return data.session_id;
};

/** 结束会话：级联清空该会话全部数据（落盘 + SQLite 数据集/分析包 + 已保存图表），释放插槽 */
export const clearData = async (sessionId: string): Promise<void> => {
  await api.post('/session/clear', null, { params: { session_id: sessionId } });
};

/* ===== 文件上传 ===== */
export const uploadFile = async (
  file: File,
  sessionId: string
): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('session_id', sessionId);
  // 不手动设置 Content-Type，让浏览器/axios 自动添加正确的 boundary
  const { data } = await api.post('/upload', formData, {
    timeout: 600000,  // 10 分钟，大文件上传需要更长时间
  });
  return data;
};

/** 上传闸门：预约数据插槽。granted=true 直接上传；false 进入排队（附 ticket_id + position） */
export const uploadGate = async (
  sessionId: string,
): Promise<{
  granted: boolean;
  session_id?: string;
  ticket_id?: string;
  position?: number;
}> => {
  const { data } = await api.post('/upload/gate', { session_id: sessionId });
  return data;
};

/** 轮询排队状态：ready（附 session_id）/ queued（附 position）/ expired */
export const getUploadQueueStatus = async (
  ticketId: string,
): Promise<{
  status: 'ready' | 'queued' | 'expired';
  session_id?: string;
  position?: number;
}> => {
  const { data } = await api.get(`/upload/queue/${ticketId}`);
  return data;
};

/** 取消排队：尽力从等待队列移除票据 */
export const cancelUploadQueue = async (ticketId: string): Promise<void> => {
  await api.post('/upload/queue/cancel', { ticket_id: ticketId });
};

/** 手动释放数据插槽：释放 df + 内存并自动晋升队首，让排队中的用户自动入队 */
export const releaseUploadSlot = async (
  sessionId: string,
): Promise<{ success: boolean; released: boolean }> => {
  const { data } = await api.post('/upload/release', { session_id: sessionId });
  return data;
};

/* ===== 数据操作 ===== */
export const getDataPreview = async (sessionId: string, rows = 100) => {
  const { data } = await api.post<PreviewResponse>('/data/preview', { session_id: sessionId, rows });
  return data;
};

export const getDataInfo = async (sessionId: string) => {
  const { data } = await api.post('/data/info', { session_id: sessionId });
  return data;
};

export const getColumnInfo = async (sessionId: string) => {
  const { data } = await api.post('/data/columns', { session_id: sessionId });
  return data;
};

export const getColumnTypes = async (sessionId: string) => {
  const { data } = await api.post('/data/column-types', { session_id: sessionId });
  return data;
};

/** 获取数据摘要统计（describe 全量指标） */
export const getDataSummary = async (sessionId: string): Promise<{ success: boolean; summary: Record<string, unknown> }> => {
  const { data } = await api.post('/data/summary', { session_id: sessionId });
  return data;
};

/* ===== 数据清洗 ===== */
export const getMissingReport = async (sessionId: string) => {
  const { data } = await api.post('/clean/missing-report', { session_id: sessionId });
  return data;
};

export const handleMissing = async (sessionId: string, column: string, method: string) => {
  const { data } = await api.post('/clean/handle-missing', { session_id: sessionId }, {
    params: { column, method },
  });
  return data;
};

export const detectTypeIssues = async (sessionId: string) => {
  const { data } = await api.post('/clean/detect-types', { session_id: sessionId });
  return data;
};

export const convertColumnType = async (sessionId: string, column: string, targetType: string) => {
  const { data } = await api.post('/clean/convert-type', { session_id: sessionId }, {
    params: { column, target_type: targetType },
  });
  return data;
};

export const detectOutliers = async (sessionId: string, method = 'iqr') => {
  const { data } = await api.post('/clean/detect-outliers', { session_id: sessionId }, {
    params: { method },
  });
  return data;
};

export const handleOutliers = async (sessionId: string, column: string, method = 'iqr', action = 'remove') => {
  const { data } = await api.post('/clean/handle-outliers', { session_id: sessionId }, {
    params: { column, method, action },
  });
  return data;
};

export const resetData = async (sessionId: string) => {
  const { data } = await api.post('/clean/reset', { session_id: sessionId });
  return data;
};

export const undoLastAction = async (sessionId: string) => {
  const { data } = await api.post('/clean/undo', { session_id: sessionId });
  return data;
};

/* ===== AI 智能清洗（异步：提交 + 轮询）===== */
export const aiClean = async (
  sessionId: string,
  request: string,
  apiKey: string,
  baseUrl?: string,
  model?: string,
  datasetIds?: string[],
): Promise<AICleanSubmitResponse> => {
  const { data } = await api.post<AICleanSubmitResponse>('/clean/ai-clean', {
    session_id: sessionId, request, api_key: apiKey, base_url: baseUrl, model, dataset_ids: datasetIds,
  });
  return data;
};

/** 轮询 AI 清洗进度（每 1.5 秒一次，直到 done/error） */
export const getAiCleanStatus = async (taskId: string): Promise<AICleanStatusResponse> => {
  const { data } = await api.get<AICleanStatusResponse>(`/clean/ai-clean/status/${taskId}`);
  return data;
};

export const getCleaningHistory = async (sessionId: string) => {
  const { data } = await api.post('/clean/history', { session_id: sessionId });
  return data;
};

/* ===== 数据导出 ===== */
export const downloadData = async (sessionId: string, original = false): Promise<Blob> => {
  const { data } = await api.post('/data/download', { session_id: sessionId, export_original: original }, { responseType: 'blob' });
  return data;
};

/** 触发 CSV 下载 */
export async function downloadCSV(sessionId: string, original = false) {
  const blob = await downloadData(sessionId, original);
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${original ? '原始' : '清洗后'}数据_${sessionId.slice(0, 8)}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

export const getCleanCompare = async (sessionId: string) => {
  const { data } = await api.post('/clean/compare', { session_id: sessionId });
  return data;
};

/* ===== AI 数据计算 ===== */
export const computeData = async (sessionId: string, query: string, apiKey: string, baseUrl?: string, model?: string) => {
  const { data } = await api.post('/data/compute', {
    session_id: sessionId, query, api_key: apiKey, base_url: baseUrl, model,
  });
  return data;
};

/* ===== 同环比专用 ===== */
export interface TbHbRow {
  month?: number;
  period: string;
  '上年值': number | null;
  '本年值': number | null;
  '同比增长率': number | null;
  '环比增长率': number | null;
}

export const getTongHuanBi = async (
  sessionId: string,
  valueColumn: string,
  dateColumn: string = '日期'
): Promise<{
  success: boolean;
  value_column: string;
  current_year: string;
  previous_year: string | null;
  rows: TbHbRow[];
  has_yoy: boolean;
  chart_option?: Record<string, unknown>;
}> => {
  const { data } = await api.post('/data/tonghuanbi', {
    session_id: sessionId,
    value_column: valueColumn,
    date_column: dateColumn,
  });
  return data;
};

/* ===== ECharts 图表 ===== */
export const createEChart = async (sessionId: string, config: ChartConfig) => {
  const { data } = await api.post<EChartResponse>('/chart/echart-create', { session_id: sessionId, ...config });
  return data;
};

/* ===== 仪表盘 ===== */
export const getDashboardKPIs = async (sessionId: string) => {
  const { data } = await api.post<KPIResponse>('/dashboard/kpis', { session_id: sessionId });
  return data;
};

/** 获取仪表盘图表（ECharts 格式） */
export const getDashboardECharts = async (sessionId: string, chartConfigs?: Record<string, unknown>[]): Promise<{ success: boolean; charts: EChartItem[] }> => {
  const { data } = await api.post('/dashboard/echarts', { session_id: sessionId, charts: chartConfigs });
  return data;
};

/* ===== 图表收藏（分析页 → 仪表盘） ===== */
export const saveChart = async (
  sessionId: string,
  title: string,
  option: Record<string, unknown>,
  chartType = '',
  tableData?: Record<string, unknown> | null,
) => {
  const { data } = await api.post('/dashboard/save-chart', {
    session_id: sessionId, title, option,
    chart_type: chartType,
    table_data: tableData || null,
  });
  return data;
};

export const getSavedCharts = async (sessionId: string): Promise<{
  success: boolean;
  charts: Array<{ title: string; option: Record<string, unknown>; saved_at: number }>;
  
}> => {
  const { data } = await api.post('/dashboard/saved-charts', { session_id: sessionId });
  return data;
};

export const deleteSavedCharts = async (sessionId: string) => {
  const { data } = await api.post('/dashboard/delete-saved-chart', { session_id: sessionId });
  return data;
};

/* ===== V2 分析包读取 ===== */
export const getSavedPackages = async (sessionId: string): Promise<{
  success: boolean;
  packages: Array<Record<string, unknown>>;
  
}> => {
  const { data } = await api.post('/dashboard/saved-packages', { session_id: sessionId });
  return data;
};

/* ===== Dashboard Schema (V2 Generator) ===== */
export const getDashboardSchema = async (
  sessionId: string,
  title?: string,
  layoutName?: string,
): Promise<{ success: boolean; schema: Record<string, unknown> }> => {
  const { data } = await api.post('/dashboard/schema', {
    session_id: sessionId,
    title: title || '',
    layout_name: layoutName || undefined,
  });
  return data;
};

/* ===== V7: Dashboard 标题 AI 命名 + 持久化 ===== */
export const generateDashboardTitle = async (
  sessionId: string,
  apiKey: string,
  baseUrl?: string,
  model?: string,
): Promise<{ success: boolean; title: string; source: 'ai' | 'fallback' }> => {
  const { data } = await api.post('/dashboard/schema/naming', {
    session_id: sessionId,
    api_key: apiKey,
    base_url: baseUrl || '',
    model: model || '',
  });
  return data;
};

export const saveDashboardTitle = async (
  sessionId: string,
  title: string,
  action: 'get' | 'set' = 'set',
): Promise<{ success: boolean; title: string; has_custom: boolean }> => {
  const { data } = await api.post('/dashboard/schema/title', {
    session_id: sessionId,
    title,
    action,
  });
  return data;
};

/* ===== 智能排版大屏（LLM 驱动） ===== */
export interface SmartLayoutRequestParams {
  session_id: string;
  api_key?: string;
  base_url?: string;
  model?: string;
  top_n?: number;
  refresh?: boolean;
}

export const getSmartLayout = async (
  params: SmartLayoutRequestParams,
): Promise<SmartLayoutResponse> => {
  const { data } = await api.post<SmartLayoutResponse>('/dashboard/smart-layout', {
    session_id: params.session_id,
    api_key: params.api_key || '',
    base_url: params.base_url || null,
    model: params.model || 'gpt-3.5-turbo',
    top_n: params.top_n ?? 12,
    refresh: params.refresh ?? false,
  });
  return data;
};

/* ===== 分析保存 ===== */
export const saveAnalysis = async (sessionId: string, packageIds: string[], datasetId?: string) => {
  const { data } = await api.post('/analysis/save', { session_id: sessionId, package_ids: packageIds, dataset_id: datasetId });
  return data;
};

/* ===== 业务推理（V3，无需 LLM）===== */
export const runReasoning = async (
  sessionId: string,
  title?: string,
): Promise<{ success: boolean; data: Record<string, unknown> }> => {
  const { data } = await api.post('/reasoning/run', {
    session_id: sessionId,
    title: title || '',
  });
  return data;
};

export const chatAnalyze = async (sessionId: string, question: string, apiKey: string, baseUrl?: string, model?: string) => {
  const { data } = await api.post<ChatResponse>('/chat/analyze', {
    session_id: sessionId, question, api_key: apiKey, base_url: baseUrl, model,
  });
  return data;
};

/** 聊天页专用：只走后端默认的 Agnes，前端不传任何模型参数。
 * choice 可选：用户点击清洗方案后回传的 method id（多轮续接）。 */
export const chatSend = async (
  sessionId: string,
  message: string,
  choice?: string,
  options?: {
    apiKey?: string | null;
    aiProvider?: string | null;
    customModel?: string | null;
    customBaseUrl?: string | null;
  },
): Promise<ChatSendResponse> => {
  const { data } = await api.post<ChatSendResponse>('/chat/send', {
    session_id: sessionId,
    message,
    choice: choice ?? null,
    api_key: options?.apiKey ?? null,
    ai_provider: options?.aiProvider ?? null,
    custom_model: options?.customModel ?? null,
    custom_base_url: options?.customBaseUrl ?? null,
  });
  return data;
};

/** 拉取会话的聊天历史（纯文字 user/assistant 对话流），用于从历史会话恢复后回填对话页。 */
export const getChatMessages = async (sessionId: string): Promise<{ success: boolean; session_id: string; messages: any[] }> => {
  const { data } = await api.get('/chat/messages', { params: { session_id: sessionId } });
  return data;
};

/* ===== 多数据集管理 ===== */

/** 切换当前分析对象（active 数据集）*/
export const selectDataset = async (sessionId: string, datasetId: string): Promise<{ success: boolean; active_dataset_id: string }> => {
  const { data } = await api.post('/data/select', { session_id: sessionId, dataset_id: datasetId });
  return data;
};

/** 拉回会话全部数据集（刷新后恢复列表）*/
export const listDatasets = async (sessionId: string): Promise<DatasetListResponse> => {
  const { data } = await api.get<DatasetListResponse>('/data/datasets', { params: { session_id: sessionId } });
  return data;
};

/** 记录会话当前所在页面（供历史恢复智能跳转） */
export const setSessionPage = async (sessionId: string, page: string): Promise<void> => {
  await api.post('/session/page', null, { params: { session_id: sessionId, page } });
};

/** 恢复指定历史会话：返回后端会话状态，前端据此切换本地会话 */
export const restoreSessionHistory = async (
  sessionId: string,
): Promise<{ session_id: string; state: Record<string, unknown>; last_page: string }> => {
  const { data } = await api.post<{ session_id: string; state: Record<string, unknown>; last_page: string }>(
    `/history/sessions/${encodeURIComponent(sessionId)}/restore`,
  );
  return data;
};

/** 保存整套 AI 配置进后端 session（刷新后随 listDatasets 一并拉回）*/
export const saveApiConfig = async (
  sessionId: string,
  cfg: { api_key: string; ai_provider: string; custom_model: string; custom_base_url: string },
): Promise<{ success: boolean }> => {
  const { data } = await api.post<{ success: boolean }>('/data/api-config', {
    session_id: sessionId,
    api_key: cfg.api_key,
    ai_provider: cfg.ai_provider,
    custom_model: cfg.custom_model,
    custom_base_url: cfg.custom_base_url,
  });
  return data;
};

/** 删除指定数据集（删落盘 + 减额度 + 回退 active）*/
export const removeDataset = async (sessionId: string, datasetId: string): Promise<{ success: boolean }> => {
  const { data } = await api.post('/data/remove-dataset', { session_id: sessionId, dataset_id: datasetId });
  return data;
};

/** 提交后台并行处理任务，返回 task_id（datasetIds 省略=全部）*/
export const processDatasets = async (
  sessionId: string,
  datasetIds?: string[],
): Promise<ProcessSubmitResponse> => {
  const { data } = await api.post<ProcessSubmitResponse>(
    `/analysis/process-datasets/${sessionId}`,
    { dataset_ids: datasetIds || null },
  );
  return data;
};

/** 轮询处理进度 */
export const getProcessStatus = async (taskId: string): Promise<ProcessStatusResponse> => {
  const { data } = await api.get<ProcessStatusResponse>(`/analysis/process-datasets/status/${taskId}`);
  return data;
};

/** 读取已落库的数据洞察分析包（process-datasets 生成结果），用于切换模块回来恢复 */
export const getDatasetPackages = async (
  sessionId: string,
  datasetId: string,
): Promise<{ packages: Record<string, any> }> => {
  const { data } = await api.get<{ packages: Record<string, any> }>(`/analysis/dataset-packages`, {
    params: { session_id: sessionId, dataset_id: datasetId },
  });
  return data;
};

/* ===== 报告 ===== */




/** V3: 从 AnalysisPackage 提取扁平指标列表 */
export const generateCards = async (sessionId: string): Promise<{
  success: boolean;
  cards: Array<Record<string, unknown>>;
  meta?: Record<string, unknown>;
  
}> => {
  const { data } = await api.post('/dashboard/cards', { session_id: sessionId });
  return data;
};

// 报告生成专用 axios 实例：不挂全局 wakeUp 重试拦截器，
// 轮询节奏与冷启动重试完全由 generateAIReport 自控，避免相互干扰。
const reportApi = axios.create({ baseURL: API_BASE, timeout: 30000 });
reportApi.interceptors.request.use((config) => {
  if (
    config.data &&
    typeof config.data === 'object' &&
    typeof (config.data as { model?: unknown }).model === 'string'
  ) {
    (config.data as { model: string }).model = (config.data as { model: string }).model.toLowerCase();
  }
  return config;
});

/**
 * 生成 AI 分析报告（异步无状态）
 *
 * 流程：提交任务（拿 task_id）→ 轮询状态 → 返回结果。
 * - 规避 Render 免费实例约 50s HTTP 超时（ERR_CONNECTION_CLOSED）。
 * - packages：前端 localStorage 中的分析包副本，后端优先使用（无状态）。
 * - 提交阶段对网络错误（冷启动）有限重试；轮询阶段对网络抖动容错继续。
 */
export const generateAIReport = async (
  sessionId: string,
  apiKey: string,
  baseUrl?: string,
  model?: string,
  packages?: Array<Record<string, unknown>>,
): Promise<AIReportResponse> => {
  // 1. 提交任务（冷启动容错：最多 3 次）
  let taskId = '';
  let lastErr: unknown = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const { data } = await reportApi.post('/report/ai-analyze', {
        session_id: sessionId,
        api_key: apiKey,
        base_url: baseUrl,
        model,
        packages,
      });
      taskId = data.task_id;
      break;
    } catch (e: unknown) {
      lastErr = e;
      const status = (e as { response?: { status?: number } })?.response?.status;
      // 业务错误（400 无分析结果 / 缺 Key 等）不重试，直接抛出
      if (status && status >= 400 && status < 500) {
        const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
        throw new Error(detail || '报告生成提交失败');
      }
      // 网络错误（可能是冷启动）→ 等待后重试
      await new Promise(r => setTimeout(r, 5000));
    }
  }
  if (!taskId) {
    const detail = (lastErr as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
    throw new Error(detail || '后端暂时无响应（可能正在冷启动），请稍后重试。');
  }

  // 2. 轮询状态（最长 5 分钟，间隔 3 秒；单次网络抖动容错继续）
  const maxWait = 300000;
  const interval = 3000;
  const start = Date.now();
  while (Date.now() - start < maxWait) {
    await new Promise(r => setTimeout(r, interval));
    let data: { status?: string; detail?: string } & Partial<AIReportResponse>;
    try {
      const resp = await reportApi.get(`/report/ai-analyze/status/${taskId}`);
      data = resp.data;
    } catch (e: unknown) {
      // 404：任务过期/进程重启 → 明确失败，提示重新生成
      if ((e as { response?: { status?: number } })?.response?.status === 404) {
        throw new Error('报告任务已过期或后端已重启，请重新生成。');
      }
      // 其它网络抖动 → 继续下一轮轮询（容错）
      continue;
    }
    if (data.status === 'done') return data as AIReportResponse;
    if (data.status === 'error') throw new Error(data.detail || '报告生成失败');
    // running → 继续轮询
  }
  throw new Error('报告生成超时（5 分钟），请重试或减少分析项后再生成。');
};

export default api;

