import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { LogOut, User as UserIcon } from 'lucide-react';

/** 右上角登录态按钮：未登录显示「登录 / 注册」；已登录显示用户名+下拉（个人中心 / 退出）。 */
const AuthButton: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, []);

  if (!user) {
    return (
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => navigate('/login')}
          className="px-4 py-1.5 rounded-full text-sm font-medium text-white shadow-md transition-all hover:scale-105"
          style={{ background: 'linear-gradient(90deg, #38BDF8 0%, #8B5CF6 100%)' }}
        >
          登录
        </button>
        <button
          type="button"
          onClick={() => navigate('/register')}
          className="px-3 py-1.5 rounded-full text-sm font-medium text-slate-700 hover:text-slate-900 hover:bg-white/70 transition-colors"
        >
          注册
        </button>
      </div>
    );
  }

  const initial = user.username.slice(0, 1).toUpperCase();

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 px-2 py-1.5 rounded-full hover:bg-white/60 transition-colors"
      >
        <span
          className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold shadow"
          style={{ background: 'linear-gradient(135deg, #38BDF8 0%, #8B5CF6 100%)' }}
        >
          {initial}
        </span>
        <span className="text-sm font-medium text-slate-700 max-w-[120px] truncate hidden md:inline">
          {user.username}
        </span>
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-44 rounded-2xl border border-white/70 bg-white/90 backdrop-blur-xl shadow-xl py-2 z-50">
          <button
            type="button"
            onClick={() => { setOpen(false); navigate('/profile'); }}
            className="w-full flex items-center gap-2 px-4 py-2 text-sm text-slate-700 hover:bg-slate-100/70 transition-colors"
          >
            <UserIcon className="w-4 h-4" /> 个人中心
          </button>
          <button
            type="button"
            onClick={() => { setOpen(false); navigate('/history'); }}
            className="w-full flex items-center gap-2 px-4 py-2 text-sm text-slate-700 hover:bg-slate-100/70 transition-colors"
          >
            <UserIcon className="w-4 h-4" /> 历史记录
          </button>
          <div className="my-1 mx-3 border-t border-slate-200/70" />
          <button
            type="button"
            onClick={async () => {
              setOpen(false);
              await logout();
              // 硬刷新整页：避免任何 in-memory React state 里残留上一个用户的 datasets/API 配置
              // 依赖 navigate 容易留下残留（UploadPage 自身的 useEffect 仍可能因旧 sessionId 再拉一次 listDatasets）。
              window.location.href = '/';
            }}
            className="w-full flex items-center gap-2 px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
          >
            <LogOut className="w-4 h-4" /> 退出登录
          </button>
        </div>
      )}
    </div>
  );
};

export default AuthButton;
