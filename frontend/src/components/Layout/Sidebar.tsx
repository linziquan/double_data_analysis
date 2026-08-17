/* Sidebar - 侧边栏导航（浅色玻璃拟态，呼应空壳） */
import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  FiGrid, FiUpload, FiCpu, FiBarChart2, FiFileText, FiSettings, FiClock,
  FiChevronLeft, FiChevronRight, FiMessageCircle,
} from 'react-icons/fi';
import { useData } from '../../contexts/DataContext';

const navItems = [
  { path: '/upload', label: '数据上传', icon: FiUpload },
  { path: '/chat', label: '智能对话', icon: FiMessageCircle },
  { path: '/models', label: 'API配置', icon: FiCpu },
  { path: '/clean', label: '数据清洗', icon: FiGrid },
  { path: '/analysis', label: '数据分析', icon: FiBarChart2 },
  { path: '/dashboard', label: '仪表盘', icon: FiFileText },
  { path: '/reports', label: 'AI报告', icon: FiFileText },
  { path: '/history', label: '历史记录', icon: FiClock },
  { path: '/settings', label: '系统设置', icon: FiSettings },
];

export default function Sidebar({
  collapsed,
  onToggle,
  mobileOpen,
  onClose,
}: {
  collapsed: boolean;
  onToggle: () => void;
  mobileOpen: boolean;
  onClose: () => void;
}) {
  const navigate = useNavigate();
  const location = useLocation();
  const { state } = useData();
  const hasData = state.rows > 0;

  const go = (path: string) => {
    navigate(path);
    onClose();
  };

  return (
    <>
      {/* 移动端遮罩：仅小屏、抽屉打开时显示 */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-30 bg-slate-900/30 backdrop-blur-sm md:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      <aside
        className={`fixed left-0 top-0 h-screen z-40 flex flex-col overflow-y-auto overflow-x-hidden pb-4 transition-[transform,width] duration-300
          ${collapsed ? 'md:w-20' : 'md:w-64'} w-64
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0`}
        style={{
          background: 'rgba(255, 255, 255, 0.55)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderRight: '1px solid rgba(255, 255, 255, 0.75)',
          boxShadow: '4px 0 24px rgba(31, 41, 55, 0.10)',
          scrollbarWidth: 'thin',
          scrollbarColor: 'rgba(100,116,139,0.35) transparent',
        }}
      >
        {/* Logo + 折叠按钮 */}
        <div className={`flex-shrink-0 flex items-center py-6 border-b border-slate-300/40 ${collapsed ? 'md:justify-center md:px-2' : 'md:px-5 md:justify-between'} px-5 justify-between`}>
          <div className={`flex items-center gap-3 ${collapsed ? 'md:hidden' : ''}`}>
            <div
              className="w-6 h-6 rounded-full flex-shrink-0"
              style={{
                background: 'radial-gradient(circle at 30% 30%, #a1c4fd 0%, #c2e9fb 30%, #ffc3a0 70%, #ffafbd 100%)',
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
              }}
            />
            {(!collapsed || mobileOpen) && (
              <div>
                <h1 className="text-lg font-bold text-slate-900 tracking-tight leading-tight">DataMind AI</h1>
                <p className="text-xs text-slate-500 mt-0.5">数据分析智能体</p>
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={() => { onToggle(); onClose(); }}
            aria-label={collapsed ? '展开侧边栏' : '收起侧边栏'}
            aria-expanded={!collapsed}
            className="p-1.5 rounded-lg text-slate-500 hover:text-slate-900 hover:bg-white/60 transition-colors flex-shrink-0"
          >
            {collapsed ? <FiChevronRight className="w-4 h-4" /> : <FiChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* 导航菜单 */}
        <nav className="flex-1 min-h-0 overflow-y-auto px-3 py-4 space-y-1">
          {navItems.map(({ path, label, icon: Icon }) => {
            const active = location.pathname === path;
            return (
              <button
                key={path}
                onClick={() => go(path)}
                title={collapsed ? label : undefined}
                aria-label={label}
                className={`w-full flex items-center gap-3 rounded-xl text-sm font-medium transition-all ${
                  collapsed ? 'md:justify-center md:px-0' : 'md:px-4'
                } px-4 py-2.5 ${
                  active
                    ? 'bg-white/70 text-slate-900 border border-violet-200 shadow-[0_4px_14px_rgba(139,92,246,0.18)]'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-white/50 border border-transparent'
                }`}
              >
                <Icon className="w-4 h-4 flex-shrink-0" style={active ? { color: '#8b5cf6' } : undefined} />
                {(!collapsed || mobileOpen) && <span>{label}</span>}
              </button>
            );
          })}
        </nav>

        {/* 底部：当前数据状态（展开时显示） */}
        {!collapsed && hasData && (
          <div className="flex-shrink-0 mx-3 px-3 py-3 rounded-xl bg-white/50 border border-slate-200/70">
            <p className="text-xs font-medium text-slate-700 truncate">{state.fileName}</p>
            <p className="text-xs text-slate-500 mt-0.5">{state.rows.toLocaleString()} 行 · {state.columns} 列</p>
          </div>
        )}
      </aside>
    </>
  );
}
