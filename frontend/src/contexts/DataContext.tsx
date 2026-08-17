/* DataMind AI - 全局数据状态管理 */
import React, { createContext, useContext, useReducer, useEffect, useCallback, ReactNode } from 'react';
import { createSession, listDatasets, setSessionPage, restoreSessionHistory } from '../api/client';
import type { ColumnInfo, DataInfo, CleaningStep } from '../types';
import { TOKEN_KEY } from '../lib/api';

// 跨 context 信号：AuthContext 登出时 dispatch 这个事件，
// DataContext 监听后清掉本设备的 sessionId 与所有 React 内存里的业务数据，
// 避免登出后仍展示上一个用户的上传文件。
// 走 window 自定义事件而不是 context 互相 import，避免循环依赖。
export const AUTH_LOGOUT_EVENT = 'datamind:auth-logout';
export const AUTH_LOGIN_EVENT = 'datamind:auth-login';

export interface AiProviderConfig {
  id: string;
  name: string;
  baseUrl: string;
  model: string;
}

export const AI_PROVIDERS: AiProviderConfig[] = [
  { id: 'ppio', name: 'PPIO 派欧云', baseUrl: 'https://api.ppio.ai/v1', model: 'deepseek-chat' },
  { id: 'deepseek', name: 'DeepSeek', baseUrl: 'https://api.deepseek.com', model: 'deepseek-chat' },
  { id: 'qwen', name: '阿里云通义千问', baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1', model: 'qwen3.7-plus' },
  { id: 'zhipu', name: '智谱 GLM', baseUrl: 'https://open.bigmodel.cn/api/paas/v4', model: 'glm-4-flash' },
  { id: 'moonshot', name: 'Moonshot / Kimi', baseUrl: 'https://api.moonshot.cn/v1', model: 'moonshot-v1-8k' },
  { id: 'openai', name: 'OpenAI', baseUrl: 'https://api.openai.com/v1', model: 'gpt-4o-mini' },
  { id: 'agnes', name: 'Agnes AI', baseUrl: 'https://apihub.agnes-ai.com/v1', model: 'agnes-2.0-flash' },
];

export interface DatasetInfo {
  dataset_id: string;
  file_name: string;
  file_size_bytes: number;
  rows: number;
  columns: string[];
  column_info: ColumnInfo[];
  preview: Record<string, unknown>[];
  uploaded_at: number;
  is_active?: boolean;
  // 多表合并宽表标记
  is_merged?: boolean;
  sources?: string[];
  merge_keys?: string[];
}

interface AnalysisState {
  tab: string;
  stats: Record<string, unknown>[] | null;
  heatmap: Record<string, unknown> | null;
  chartFigure: Record<string, unknown> | null;
  chartType: string;
  chartX: string;
  chartY: string;
  chatHistory: { role: string; content: string }[];
  insights: string;
  quickInsights: string[];
  computeResult: string;
  savedCount: number;
}

interface DataState {
  sessionId: string;
  fileName: string;
  rows: number;
  columns: number;
  preview: Record<string, unknown>[];
  columnInfo: ColumnInfo[];
  dataInfo: DataInfo | null;
  cleaningHistory: CleaningStep[];
  apiKey: string;
  aiProvider: string;  // ID of selected AI provider
  customModel: string;  // 用户自定义模型名（为空则用服务商预设默认值，如 qwen-plus → qwen-max）
  customBaseUrl: string;  // 用户自定义 API 地址（为空则用服务商预设 baseUrl，用于百炼新版等需要 WorkspaceId 的场景）
  loading: boolean;
  error: string | null;
  analysis: AnalysisState;
  // ===== 多数据集管理（顶层 fileName/rows/columns/preview/columnInfo 始终代表 active 数据集，下游零改）=====
  datasets: DatasetInfo[];
  activeDatasetId: string | null;
  usedBytes: number;
  quotaBytes: number;
  /** 当前用户已上传的数据集数量（登录态才有意义） */
  datasetCount: number;
  /** 单用户数据集上限（登录态才有，游客为 null） */
  datasetLimit: number | null;
}

type Action =
  | { type: 'SET_SESSION'; sessionId: string }
  | { type: 'SET_DATA'; payload: Partial<DataState> }
  | { type: 'SET_PREVIEW'; preview: Record<string, unknown>[] }
  | { type: 'SET_CLEANING_HISTORY'; history: CleaningStep[] }
  | { type: 'SET_API_KEY'; apiKey: string }
  | { type: 'SET_AI_PROVIDER'; aiProvider: string }
  | { type: 'SET_CUSTOM_MODEL'; customModel: string }
  | { type: 'SET_CUSTOM_BASE_URL'; customBaseUrl: string }
  | { type: 'SET_API_CONFIG'; apiKey: string; aiProvider: string; customModel: string; customBaseUrl: string }
  | { type: 'SET_LOADING'; loading: boolean }
  | { type: 'SET_ERROR'; error: string | null }
  | { type: 'SET_ANALYSIS'; payload: Partial<AnalysisState> }
  | { type: 'CLEAR_DATA' }
  | { type: 'ADD_DATASET'; payload: DatasetInfo }
  | { type: 'SELECT_DATASET'; datasetId: string }
  | { type: 'REMOVE_DATASET'; datasetId: string }
  | { type: 'SET_DATASETS'; datasets: DatasetInfo[] }
  | { type: 'SET_QUOTA'; usedBytes: number; quotaBytes: number; datasetCount?: number; datasetLimit?: number | null };

const initialAnalysis: AnalysisState = {
  tab: 'stats',
  stats: null,
  heatmap: null,
  chartFigure: null,
  chartType: 'bar',
  chartX: '',
  chartY: '',
  chatHistory: [],
  insights: '',
  quickInsights: [],
  computeResult: '',
  savedCount: 0,
};

const initialState: DataState = {
  sessionId: '',
  fileName: '',
  rows: 0,
  columns: 0,
  preview: [],
  columnInfo: [],
  dataInfo: null,
  cleaningHistory: [],
  apiKey: '',
  aiProvider: 'ppio',
  customModel: '',
  customBaseUrl: '',
  loading: false,
  error: null,
  analysis: initialAnalysis,
  datasets: [],
  activeDatasetId: null,
  usedBytes: 0,
  quotaBytes: 0,
  datasetCount: 0,
  datasetLimit: null,
};

// 把某个数据集的字段回放到顶层（fileName/rows/columns/preview/columnInfo）
// 无 active 数据集时清空顶层，避免删除/刷新后残留旧数据
function replayToTop(state: DataState, ds: DatasetInfo | undefined): Partial<DataState> {
  if (!ds) return { fileName: '', rows: 0, columns: 0, preview: [], columnInfo: [] };
  return {
    fileName: ds.file_name,
    rows: ds.rows,
    columns: ds.columns.length,
    preview: ds.preview,
    columnInfo: ds.column_info,
  };
}

function dataReducer(state: DataState, action: Action): DataState {
  switch (action.type) {
    case 'SET_SESSION':
      return { ...state, sessionId: action.sessionId };
    case 'SET_DATA':
      return { ...state, ...action.payload };
    case 'SET_PREVIEW':
      return { ...state, preview: action.preview };
    case 'SET_CLEANING_HISTORY':
      return { ...state, cleaningHistory: action.history };
    case 'SET_API_KEY':
      return { ...state, apiKey: action.apiKey };
    case 'SET_AI_PROVIDER':
      return { ...state, aiProvider: action.aiProvider, customModel: '', customBaseUrl: '' };  // 切换服务商时清空自定义配置
    case 'SET_CUSTOM_MODEL':
      return { ...state, customModel: action.customModel };
    case 'SET_CUSTOM_BASE_URL':
      return { ...state, customBaseUrl: action.customBaseUrl };
    case 'SET_API_CONFIG':
      // 刷新回放：一次性把后端 session 里的四个 API 字段写回 state（不走"切换即清空"逻辑）
      return {
        ...state,
        apiKey: action.apiKey,
        aiProvider: action.aiProvider,
        customModel: action.customModel,
        customBaseUrl: action.customBaseUrl,
      };
    case 'SET_LOADING':
      return { ...state, loading: action.loading };
    case 'SET_ERROR':
      return { ...state, error: action.error };
    case 'SET_ANALYSIS':
      return { ...state, analysis: { ...state.analysis, ...action.payload } };
    case 'ADD_DATASET': {
      const datasets = [...state.datasets, action.payload];
      // 仅当该数据集显式 is_active（多 sheet 拆分的首个 sheet）或列表原本为空时才切换 active，
      // 避免批量上传多 sheet 时 active 被逐个覆盖为最后一个
      const makeActive = action.payload.is_active === true || state.datasets.length === 0;
      const activeDatasetId = makeActive ? action.payload.dataset_id : state.activeDatasetId;
      const activeDs = makeActive ? action.payload : (datasets.find(d => d.dataset_id === activeDatasetId) || null);
      return {
        ...state,
        datasets,
        activeDatasetId,
        ...replayToTop(state, activeDs),
      };
    }
    case 'SELECT_DATASET': {
      const targetId = action.datasetId;
      const datasets = state.datasets.map(d => ({
        ...d,
        is_active: d.dataset_id === targetId,
      }));
      const ds = datasets.find(d => d.dataset_id === targetId);
      return {
        ...state,
        datasets,
        activeDatasetId: targetId,
        ...replayToTop(state, ds),
      };
    }
    case 'REMOVE_DATASET': {
      const datasets = state.datasets.filter(d => d.dataset_id !== action.datasetId);
      let next = { ...state, datasets, activeDatasetId: state.activeDatasetId };
      if (state.activeDatasetId === action.datasetId) {
        const nextActive = datasets[0];
        next = {
          ...next,
          activeDatasetId: nextActive ? nextActive.dataset_id : null,
          ...replayToTop(state, nextActive),
        };
      }
      return next;
    }
    case 'SET_DATASETS': {
      // 刷新拉回：替换列表。优先用后端标记的 is_active 挑 active；
      // 若本地已知 activeDatasetId 且它仍在列表里，则以本地为准（兼容多 sheet 切换场景）。
      const list = Array.isArray(action.datasets) ? action.datasets : [];
      const localActive = state.activeDatasetId
        ? list.find(d => d.dataset_id === state.activeDatasetId)
        : undefined;
      const ds = localActive || list.find(d => d.is_active) || undefined;
      const activeDatasetId = ds ? ds.dataset_id : state.activeDatasetId;
      return {
        ...state,
        datasets: list,
        activeDatasetId,
        ...replayToTop(state, ds),
      };
    }
    case 'SET_QUOTA':
      return {
        ...state,
        usedBytes: action.usedBytes,
        quotaBytes: action.quotaBytes,
        datasetCount: action.datasetCount ?? state.datasetCount,
        datasetLimit: action.datasetLimit === undefined ? state.datasetLimit : action.datasetLimit,
      };
    case 'CLEAR_DATA':
      // 结束会话：清掉 localStorage 里的旧 sessionId，让 ensureValidSession 自然创建全新会话，
      // 避免沿用「已被后端 clear_data 删除」的旧 id（否则 ensureValidSession 直接返回旧 id、新会话语义失效）。
      // API 配置（apiKey/aiProvider/customModel/customBaseUrl）与数据集同生命周期，结束会话一并清空，不再跨会话保留。
      localStorage.removeItem('sessionId');
      return { ...initialState };
    default:
      return state;
  }
}

interface DataContextType {
  state: DataState;
  dispatch: React.Dispatch<Action>;
  /** 确保当前有有效 sessionId：有则保留，无则创建新会话并写入 localStorage。
   *  不在初始化 effect 中盲目换新，避免死守旧 id 或误换新 id。 */
  ensureValidSession: () => Promise<string>;
  /** 记录会话当前所在页面（供历史恢复智能跳转）。 */
  setCurrentPage: (page: string) => void;
  /** 从「历史会话」一键恢复：切换本地会话到目标 session 并重建全部数据。 */
  restoreSession: (sessionId: string) => Promise<void>;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

export function DataProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(dataReducer, initialState);

  // 确保当前有有效 sessionId：有则保留，无则创建新会话并写入 localStorage。
  // 注意：不在挂载时盲目换新 id，仅当 localStorage 无记录时才创建（避免误丢数据）。
  const ensureValidSession = useCallback(async (): Promise<string> => {
    const existing = state.sessionId || localStorage.getItem('sessionId');
    if (existing) {
      if (!state.sessionId) dispatch({ type: 'SET_SESSION', sessionId: existing });
      return existing;
    }
    const sid = await createSession();
    dispatch({ type: 'SET_SESSION', sessionId: sid });
    localStorage.setItem('sessionId', sid);
    return sid;
  }, [state.sessionId, dispatch]);

  // 拉回会话全部数据集 + API 配置，并重建 active。供初始化与历史恢复复用。
  const loadSessionData = useCallback(async (sid: string) => {
    const res = await listDatasets(sid);
    dispatch({ type: 'SET_DATASETS', datasets: res.datasets });
    const active = res.datasets.find((d) => d.is_active);
    if (active) dispatch({ type: 'SELECT_DATASET', datasetId: active.dataset_id });
    dispatch({
      type: 'SET_API_CONFIG',
      apiKey: res.api_key ?? '',
      aiProvider: res.ai_provider ?? '',
      customModel: res.custom_model ?? '',
      customBaseUrl: res.custom_base_url ?? '',
    });
  }, [dispatch]);

  // 初始化：从 localStorage 恢复 session 或创建新 session，并全局拉回已上传数据集，
  // 使任意页面（仪表盘/分析/清洗/AI报告）刷新后无需先进入上传页即可拿到 hasData 所需数据。
  useEffect(() => {
    let alive = true;
    (async () => {
      const sid = await ensureValidSession();
      if (!alive || !sid) return;
      try {
        await loadSessionData(sid);
      } catch {
        // 会话暂无数据集，忽略；任由页面展示空态
      }
    })();
    return () => { alive = false; };
  }, [ensureValidSession, loadSessionData]);

  // 记录当前页面（志愿者/已登录均可），供历史恢复时智能跳转。
  const setCurrentPage = useCallback((page: string) => {
    if (!page) return;
    const sid = state.sessionId || localStorage.getItem('sessionId');
    if (sid) {
      // 不阻塞主流程：失败静默忽略（如未登录态不可恢复）
      setSessionPage(sid, page).catch(() => {});
    }
  }, [state.sessionId]);

  // 从历史记录恢复会话：后端校验归属后，前端切换本地 sessionId 并重建数据。
  const restoreSession = useCallback(async (sessionId: string) => {
    await restoreSessionHistory(sessionId);
    localStorage.setItem('sessionId', sessionId);
    dispatch({ type: 'SET_SESSION', sessionId });
    try {
      await loadSessionData(sessionId);
    } catch {
      // 忽略：恢复后页面随 last_page 跳转，空态也可接受
    }
  }, [dispatch, loadSessionData]);

  // 监听 AuthContext 派发的登出事件：清掉本设备 sessionId 与 React 内存中的全部业务数据，
  // 防止「登出后页面仍展示上一个用户上传的文件」。
  // 注意：登出后不要立即创建新 session，否则会被后续登录误当作"游客 session"回填成一条
  // 0 数据 0 分析包的空历史记录。游客若继续操作（如进入上传页），会在用到时通过
  // ensureValidSession 自然创建，且不会被加入历史（user_id 仍为 NULL）。
  useEffect(() => {
    const onLogout = () => {
      dispatch({ type: 'CLEAR_DATA' });
    };
    window.addEventListener(AUTH_LOGOUT_EVENT, onLogout);
    return () => window.removeEventListener(AUTH_LOGOUT_EVENT, onLogout);
  }, [dispatch]);

  // 兜底守卫：监听 localStorage 中 token 的变化（同标签页 + 跨标签页）。
  // AuthContext 在某些 race 场景下可能只删 token 不派发 AUTH_LOGOUT_EVENT
  // （例如初始恢复时 refresh 失败、行 48-50 仅 setToken(null) 不派事件）。
  // 这里一旦检测到"本设备 token 缺失"且 datasets 仍非空，立即强制 CLEAR_DATA，
  // 避免未登录态仍显示已登录用户的 session 数据。
  // 重要：必须用 lib/api 的 TOKEN_KEY（'datamind_token'）而不是字符串 'token'，
  // 否则这里永远监听不到 AuthContext 的增删 → 未登录态仍残留旧数据。
  useEffect(() => {
    let prevToken = typeof window !== 'undefined' ? localStorage.getItem(TOKEN_KEY) : null;
    let didDispatch = false;
    const check = () => {
      const cur = localStorage.getItem(TOKEN_KEY);
      if (prevToken && !cur && !didDispatch) {
        // 从有 token → 无 token：等价于本设备登出
        dispatch({ type: 'CLEAR_DATA' });
        didDispatch = true;
      }
      if (cur) didDispatch = false; // 重新出现 token 时复位，等待下次"有→无"再触发
      prevToken = cur;
    };
    // 跨标签页 storage 事件
    window.addEventListener('storage', check);
    // 同标签页轮询兜底（每 500ms 检查一次；额外开销可忽略）
    const tick = window.setInterval(check, 500);
    return () => {
      window.removeEventListener('storage', check);
      window.clearInterval(tick);
    };
  }, [dispatch]);

  // 启动时一次性检查：如果初次进入页面时 token 就已为空（例如刷新后丢了登录态）
  // 但 state.datasets 仍非空，则主动清空，避免把上一个会话的残留显示给登出后的用户。
  useEffect(() => {
    const t = typeof window !== 'undefined' ? localStorage.getItem(TOKEN_KEY) : null;
    if (!t && state.datasets.length > 0) {
      dispatch({ type: 'CLEAR_DATA' });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 监听 AuthContext 派发的登录事件：把后端分配的 sessionId 同步进 React state。
  // 修复「登录后 React state.sessionId 与 localStorage.sessionId 不一致」的 bug
  // （此前登出清掉 localStorage 重建新游客 session 后，登录时回填它，回到空的内存里，
  // 再加上历史记录看到的就是这个无数据的 session）。
  useEffect(() => {
    const onLogin = (ev: Event) => {
      const detail = (ev as CustomEvent).detail as { sessionId?: string } | undefined;
      const sid = detail?.sessionId;
      if (sid && sid !== state.sessionId) {
        dispatch({ type: 'SET_SESSION', sessionId: sid });
      }
    };
    window.addEventListener(AUTH_LOGIN_EVENT, onLogin);
    return () => window.removeEventListener(AUTH_LOGIN_EVENT, onLogin);
  }, [dispatch, state.sessionId]);

  return (
    <DataContext.Provider value={{ state, dispatch, ensureValidSession, setCurrentPage, restoreSession }}>
      {children}
    </DataContext.Provider>
  );
}

export function useData(): DataContextType {
  const ctx = useContext(DataContext);
  if (!ctx) throw new Error('useData must be used within DataProvider');
  return ctx;
}
