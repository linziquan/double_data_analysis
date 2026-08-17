import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { History, ChevronRight, Clock, AlertCircle, Layers, FileBarChart, Trash2, MessageSquare } from 'lucide-react';
import { listSessions, deleteHistorySession, type HistorySession } from '../lib/api';
import { useData } from '../contexts/DataContext';

// 会话"最后访问页面"到前端路由的映射（智能恢复上下文时用）
const PAGE_ROUTE: Record<string, string> = {
  chat: '/chat',
  upload: '/upload',
  clean: '/clean',
  analysis: '/analysis',
  dashboard: '/dashboard',
  reports: '/reports',
  models: '/models',
  settings: '/settings',
  profile: '/profile',
};
const safeRoute = (lastPage: string): string => PAGE_ROUTE[lastPage] || '/upload';

export default function HistoryPage() {
  const navigate = useNavigate();
  const { restoreSession, state, ensureValidSession } = useData();
  const [sessions, setSessions] = useState<HistorySession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [restoringId, setRestoringId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const list = await listSessions();
      setSessions(Array.isArray(list) ? list : []);
    } catch (err: any) {
      setError(err?.message || '加载历史会话失败');
      setSessions([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 点开历史会话：恢复数据 + 智能跳转到智能对话页（统一入口，绕过 last_page）
  // 之前的逻辑会按 last_page 跳转（如 /upload），导致对话上下文无法快速续上；
  // 现在统一进入 /chat，由 ChatPage 的 useEffect 自动回填 session.messages，
  // 即使用户上次最后停留在数据上传页，也能直接看到历史对话记录。
  const onOpen = async (session: HistorySession) => {
    if (restoringId || deletingId) return;
    setRestoringId(session.session_id);
    setError('');
    try {
      await restoreSession(session.session_id);
      navigate('/chat', { replace: true });
    } catch (err: any) {
      setError(err?.message || '恢复会话失败');
      setRestoringId(null);
    }
  };

  // 删除会话：带确认 + 禁用误删当前激活会话（防止登出后被孤立在空 state 上）
  const onDelete = async (session: HistorySession) => {
    if (restoringId || deletingId) return;
    const isCurrent = state.sessionId === session.session_id;
    const ok = window.confirm(
      isCurrent
        ? `当前激活的会话"${session.title}"将被永久删除。\n删除后页面会清空并为你分配一个全新的空会话，确定继续吗？`
        : `确认删除会话"${session.title}"吗？该会话的数据集与分析包将被永久清除，且无法恢复。`
    );
    if (!ok) return;
    setDeletingId(session.session_id);
    setError('');
    try {
      await deleteHistorySession(session.session_id);
      // 乐观更新：直接从本地列表中移除
      setSessions((prev) => prev.filter((s) => s.session_id !== session.session_id));
      // 删的是当前激活的会话：清理本地状态 + 触发新会话分配（与登录路径一致）
      if (isCurrent) {
        localStorage.removeItem('sessionId');
        // ensureValidSession 内部会创建新 session 并 SET_SESSION 同步 React state
        try { await ensureValidSession(); } catch { /* ignore */ }
      }
    } catch (err: any) {
      setError(err?.message || '删除失败');
    } finally {
      setDeletingId(null);
    }
  };

  const fmtTime = (t?: number) =>
    t ? new Date(t * 1000).toLocaleString('zh-CN', { hour12: false }) : '—';

  return (
    <div className="max-w-4xl mx-auto page-enter">
      <h1 className="text-2xl font-bold text-slate-900 mb-1">历史会话</h1>
      <p className="text-sm text-slate-500 mb-4">
        记录了你每次的分析过程，点开即可回到上次的进度，无需重复上传
        <span className="text-slate-400">（每个账号最多保留最近 10 条，最旧的会自动清理）</span>
      </p>

      {error && (
        <div className="flex items-center gap-2 mb-4 px-3 py-2 rounded-xl bg-red-50 text-red-600 text-sm">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {loading ? (
        <p className="text-slate-400 text-sm">加载中…</p>
      ) : sessions.length === 0 ? (
        <div className="glass-card rounded-2xl p-10 text-center text-slate-400">
          <History className="w-10 h-10 mx-auto mb-3 text-slate-300" />
          暂无历史会话，去上传分析你的第一份数据吧～
        </div>
      ) : (
        <div className="space-y-3">
          {sessions.map((s) => {
            const isRestoring = restoringId === s.session_id;
            const isDeleting = deletingId === s.session_id;
            const isCurrent = state.sessionId === s.session_id;
            return (
              <div
                key={s.session_id}
                className={`glass-card rounded-2xl w-full px-4 py-3.5 flex items-center gap-3
                            hover:shadow-md hover:border-violet-200 transition-all
                            ${(isRestoring || isDeleting) ? 'opacity-60' : ''}
                            group`}
              >
                <button
                  type="button"
                  onClick={() => onOpen(s)}
                  disabled={!!isRestoring || !!isDeleting}
                  className="flex items-center gap-3 flex-1 min-w-0 text-left disabled:cursor-wait"
                >
                  <div className="p-2 rounded-xl bg-violet-50 text-violet-600 group-hover:bg-violet-100 transition-colors">
                    {isRestoring ? (
                      <span className="block w-5 h-5 rounded-full border-2 border-violet-400 border-t-transparent animate-spin" />
                    ) : (
                      <History className="w-5 h-5" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-slate-900 truncate flex items-center gap-2">
                      {s.title}
                      {isCurrent && (
                        <span className="text-[10px] font-semibold text-violet-600 bg-violet-50 border border-violet-200 rounded-full px-1.5 py-0.5">
                          当前
                        </span>
                      )}
                    </p>
                    <div className="flex items-center gap-3 mt-0.5 text-xs text-slate-400">
                      <span className="flex items-center gap-1">
                        <Layers className="w-3.5 h-3.5" />
                        {s.dataset_count} 数据集
                      </span>
                      <span className="flex items-center gap-1">
                        <FileBarChart className="w-3.5 h-3.5" />
                        {s.package_count} 分析包
                      </span>
                      {Number(s.chat_count || 0) > 0 && (
                        <span className="flex items-center gap-1 text-violet-500">
                          <MessageSquare className="w-3.5 h-3.5" />
                          {s.chat_count} 条对话
                        </span>
                      )}
                      <span className="flex items-center gap-1">
                        <Clock className="w-3.5 h-3.5" />
                        {fmtTime(s.last_access)}
                      </span>
                    </div>
                  </div>
                  <span className="text-xs text-slate-400 hidden sm:inline">
                    {isRestoring ? '恢复中…' : '继续'}
                  </span>
                  <ChevronRight className="w-5 h-5 text-slate-400 group-hover:text-violet-500 transition-colors" />
                </button>
                {/* 删除按钮：用 div + onClick 包裹避免内嵌 button（HTML 不允许嵌套） */}
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); onDelete(s); }}
                  disabled={!!isRestoring || !!isDeleting}
                  className="ml-1 p-2 rounded-xl text-slate-400 hover:text-red-500 hover:bg-red-50
                             disabled:opacity-50 disabled:cursor-not-allowed transition-colors
                             flex items-center gap-1"
                  title="删除该会话"
                  aria-label="删除该会话"
                >
                  {isDeleting ? (
                    <span className="block w-4 h-4 rounded-full border-2 border-red-300 border-t-transparent animate-spin" />
                  ) : (
                    <Trash2 className="w-4 h-4" />
                  )}
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
