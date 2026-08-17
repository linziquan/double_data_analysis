import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Lock, AlertCircle, CheckCircle2, LogOut } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { authChangePassword, authLogout } from '../lib/api';

export default function ProfilePage() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [oldPwd, setOldPwd] = useState('');
  const [newPwd, setNewPwd] = useState('');
  const [confirmPwd, setConfirmPwd] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    if (newPwd.length < 6) {
      setError('新密码至少 6 位');
      return;
    }
    if (newPwd !== confirmPwd) {
      setError('两次输入的新密码不一致');
      return;
    }
    setLoading(true);
    try {
      await authChangePassword(oldPwd, newPwd);
      setSuccess('密码已修改，请重新登录');
      setOldPwd('');
      setNewPwd('');
      setConfirmPwd('');
      // 改密后端已 token_version+1 使旧 token 失效，前端需重新登录
      setTimeout(async () => {
        await logout();
        navigate('/login');
      }, 1200);
    } catch (err: any) {
      setError(err?.message || '修改失败');
    } finally {
      setLoading(false);
    }
  };

  const onLogout = async () => {
    await logout();
    // 硬刷新整页：避免任何 in-memory React state 里残留上一个用户的 datasets/API 配置
    // 依赖 navigate 容易留下残留（UploadPage 自身的 useEffect 仍可能因旧 sessionId 再拉一次 listDatasets）。
    window.location.href = '/';
  };

  return (
    <div className="max-w-2xl mx-auto page-enter">
      <h1 className="text-2xl font-bold text-slate-900 mb-6">个人中心</h1>

      <div className="glass-card rounded-2xl p-6 mb-6">
        <p className="text-sm text-slate-500">当前账户</p>
        <div className="flex items-center gap-3 mt-2">
          <span
            className="w-12 h-12 rounded-full flex items-center justify-center text-white text-xl font-bold shadow"
            style={{ background: 'linear-gradient(135deg, #38BDF8 0%, #8B5CF6 100%)' }}
          >
            {user?.username.slice(0, 1).toUpperCase()}
          </span>
          <div>
            <p className="text-lg font-semibold text-slate-900">{user?.username}</p>
            <p className="text-xs text-slate-400">ID: {user?.id}</p>
          </div>
        </div>
      </div>

      <form onSubmit={onSubmit} className="glass-card rounded-2xl p-6 mb-6 space-y-4">
        <h2 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
          <Lock className="w-5 h-5 text-violet-600" /> 修改密码
        </h2>

        {error && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-red-50 text-red-600 text-sm">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}
        {success && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-emerald-50 text-emerald-600 text-sm">
            <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
            <span>{success}</span>
          </div>
        )}

        <input
          type="password"
          value={oldPwd}
          onChange={(e) => setOldPwd(e.target.value)}
          placeholder="原密码"
          className="glass-input w-full px-4 py-3 text-slate-800"
        />
        <input
          type="password"
          value={newPwd}
          onChange={(e) => setNewPwd(e.target.value)}
          placeholder="新密码（至少 6 位）"
          className="glass-input w-full px-4 py-3 text-slate-800"
        />
        <input
          type="password"
          value={confirmPwd}
          onChange={(e) => setConfirmPwd(e.target.value)}
          placeholder="确认新密码"
          className="glass-input w-full px-4 py-3 text-slate-800"
        />
        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 rounded-xl text-white font-semibold shadow-lg transition-all hover:scale-[1.02] disabled:opacity-60"
          style={{ background: 'linear-gradient(90deg, #38BDF8 0%, #8B5CF6 100%)' }}
        >
          {loading ? '保存中…' : '保存修改'}
        </button>
      </form>

      <button
        type="button"
        onClick={onLogout}
        className="w-full flex items-center justify-center gap-2 py-3 rounded-xl text-red-600 border border-red-200 bg-white/60 hover:bg-red-50 transition-colors font-medium"
      >
        <LogOut className="w-5 h-5" /> 退出登录
      </button>
    </div>
  );
}
