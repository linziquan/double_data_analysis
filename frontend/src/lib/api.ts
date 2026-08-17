/* DataMind AI - Auth API 客户端（基于 axios 的独立实例，仅用于认证相关请求） */
import axios from 'axios';

export const TOKEN_KEY = 'datamind_token';
export const REFRESH_TOKEN_KEY = 'datamind_refresh_token';
export const REDIRECT_KEY = 'datamind_post_login_redirect';

// 与 client.ts 一致的 baseURL 解析规则
let API_BASE = import.meta.env.VITE_API_BASE || '/api';
API_BASE = API_BASE.replace(/\/$/, '');
if (!API_BASE.endsWith('/api')) API_BASE += '/api';

export const authApi = axios.create({
  baseURL: API_BASE,
  timeout: 20000,
});

// 请求拦截：注入 Authorization
authApi.interceptors.request.use((config) => {
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// 续期锁：避免并发 401 触发多次 refresh
let refreshing: Promise<string | null> | null = null;

export const refreshAuth = async (): Promise<string | null> => {
  const rt = localStorage.getItem(REFRESH_TOKEN_KEY);
  if (!rt) return null;
  try {
    const { data } = await authApi.post('/auth/refresh', { refresh_token: rt });
    const newAccess = data.token as string;
    localStorage.setItem(TOKEN_KEY, newAccess);
    return newAccess;
  } catch {
    // refresh 失败（refresh token 过期/被吊销）：清本地态，交由拦截器跳登录
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    return null;
  }
};

// 统一续期入口：带锁，保证并发请求只刷一次
export const tryRefresh = async (): Promise<string | null> => {
  if (!refreshing) {
    refreshing = refreshAuth().finally(() => {
      refreshing = null;
    });
  }
  return refreshing;
};

// 响应拦截：401 → 先尝试 refresh 续期并重试一次；失败则记录来源跳 /login
authApi.interceptors.response.use(
  (res) => res,
  async (err) => {
    const status = err.response?.status;
    const url = err.config?.url || '';
    const isAuthCall = url.includes('/auth/');
    if (status === 401 && !isAuthCall) {
      // 业务接口 401：尝试用 refresh token 续期后重试一次
      const newToken = await tryRefresh();
      if (newToken && err.config) {
        err.config.headers.Authorization = `Bearer ${newToken}`;
        return authApi.request(err.config);
      }
      // 续期失败：记录来源并跳登录
      const current = window.location.pathname + window.location.search + window.location.hash;
      sessionStorage.setItem(REDIRECT_KEY, current);
      if (!window.location.pathname.startsWith('/login') &&
          !window.location.pathname.startsWith('/register')) {
        window.location.href = '/login';
      }
    }
    let msg = err.response?.data?.detail || err.message || '请求失败';
    if (typeof msg !== 'string') msg = String(msg);
    return Promise.reject(new Error(msg));
  },
);

/* ===== 认证接口 ===== */
export const authRegister = async (username: string, password: string, sessionId?: string) => {
  const { data } = await authApi.post('/auth/register', { username, password, session_id: sessionId });
  // 后端登录/注册响应包含 session_id 字段，供前端立即覆盖 localStorage.sessionId，
  // 避免游客 session 残留（user_id=NULL），下次冷启动被 clear_guest_data 误删。
  return data as {
    token: string; refresh_token: string;
    user: { id: number; username: string };
    session_id?: string;
  };
};

export const authLogin = async (username: string, password: string, sessionId?: string) => {
  const { data } = await authApi.post('/auth/login', { username, password, session_id: sessionId });
  return data as {
    token: string; refresh_token: string;
    user: { id: number; username: string };
    session_id?: string;
  };
};

export const authMe = async () => {
  const { data } = await authApi.get('/auth/me');
  return data as { id: number; username: string };
};

export const authChangePassword = async (old_password: string, new_password: string) => {
  const { data } = await authApi.post('/auth/change-password', { old_password, new_password });
  return data as { success: boolean; token_version: number };
};

export const authLogout = async () => {
  const { data } = await authApi.post('/auth/logout');
  return data as { success: boolean };
};

/* ===== 历史接口（按用户归集，需登录）===== */
export const listUserDatasets = async () => {
  const { data } = await authApi.get('/history/datasets');
  return data as { datasets: Array<Record<string, any>> };
};

/** 列出当前用户的全部历史会话（会话维度），用于"历史记录"页 */
export interface HistorySession {
  session_id: string;
  title: string;
  last_page: string;
  dataset_count: number;
  package_count: number;
  /** 该会话中累计的对话轮次（user+assistant 累计）。0 表示只纯数据集/分析包。 */
  chat_count?: number;
  created_at: number;
  last_access: number;
}
export const listSessions = async (): Promise<HistorySession[]> => {
  const { data } = await authApi.get('/history/sessions');
  return (data?.sessions as HistorySession[]) || [];
};

export const listUserPackages = async (datasetId: string) => {
  const { data } = await authApi.get('/history/packages', { params: { dataset_id: datasetId } });
  return data as { packages: Array<Record<string, any>> };
};

export const deleteUserDataset = async (datasetId: string) => {
  const { data } = await authApi.delete(`/history/dataset/${datasetId}`);
  return data as { success: boolean };
};

export const deleteUserPackage = async (packageId: string) => {
  const { data } = await authApi.delete(`/history/package/${packageId}`);
  return data as { success: boolean };
};

/** 删除指定历史会话（仅本人会话；后端已做归属校验）。 */
export const deleteHistorySession = async (sessionId: string) => {
  const { data } = await authApi.delete(`/history/sessions/${sessionId}`);
  return data as { success: boolean };
};

/* ===== P2：收藏 / 分组 ===== */
export interface FavoriteGroup {
  group_name: string;
  items: Array<{
    package_id: string;
    is_starred: boolean;
    display_name: string | null;
    sort_order: number;
    title: string;
    package_type: string;
    created_at: number | null;
    saved_at: string | null;
  }>;
}

export const listFavorites = async () => {
  const { data } = await authApi.get('/history/favorites');
  return data as { groups: FavoriteGroup[] };
};

export const toggleFavorite = async (packageId: string, starred: boolean) => {
  const { data } = await authApi.post('/history/favorites/toggle', {
    package_id: packageId,
    starred,
  });
  return data as { success: boolean; state: Record<string, any> };
};

export const updateFavoriteMeta = async (
  packageId: string,
  meta: { display_name?: string; group_name?: string; sort_order?: number },
) => {
  const { data } = await authApi.post('/history/favorites/meta', {
    package_id: packageId,
    ...meta,
  });
  return data as { success: boolean; state: Record<string, any> };
};

/* ===== P2：分享链接 ===== */
export const createShare = async (packageId: string, expireAt?: number | null) => {
  const { data } = await authApi.post('/history/shares', {
    package_id: packageId,
    expire_at: expireAt ?? null,
  });
  return data as { success: boolean; share_id: string; package_id: string; expire_at: number | null };
};

export const listMyShares = async () => {
  const { data } = await authApi.get('/history/shares');
  return data as { shares: Array<Record<string, any>> };
};

export const deleteShare = async (shareId: string) => {
  const { data } = await authApi.delete(`/history/share/${shareId}`);
  return data as { success: boolean };
};

// 公开只读：无需登录，使用独立 GET（authApi 会自动带 token，但该端点无鉴权，不影响）
export const getSharedPackage = async (shareId: string) => {
  const { data } = await authApi.get(`/history/shared/${shareId}`);
  return data as {
    share_id: string;
    package_id: string;
    created_at: number;
    expire_at: number | null;
    payload: Record<string, any> | null;
  };
};
