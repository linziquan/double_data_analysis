/* DataMind AI - 应用入口与路由 */
import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { DataProvider } from './contexts/DataContext';
import { AuthProvider } from './context/AuthContext';
import Layout from './components/Layout/Layout';
import RequireAuth from './components/RequireAuth';
import UploadPage from './pages/UploadPage';
import CleanPage from './pages/CleanPage';
import AnalysisPage from './pages/AnalysisPage';
import DashboardPage from './pages/DashboardPage';
import EtherealPreview from './EtherealPreview';
import AIModelsPage from './pages/AIModelsPage';
import ReportsPage from './pages/ReportsPage';
import SettingsPage from './pages/SettingsPage';
import CoverPage from './pages/CoverPage';
import ChatPage from './pages/ChatPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ProfilePage from './pages/ProfilePage';
import HistoryPage from './pages/HistoryPage';
import SharedPage from './pages/SharedPage';
import { useData } from './contexts/DataContext';
import { useLocation } from 'react-router-dom';

// 记录会话当前所在页面，供历史恢复时智能跳转（排除登录/注册/封面/历史页自身）
function PageTracker() {
  const location = useLocation();
  const { setCurrentPage } = useData();
  React.useEffect(() => {
    const p = location.pathname;
    if (p === '/login' || p === '/register' || p === '/' || p === '/history') return;
    setCurrentPage(p);
  }, [location.pathname, setCurrentPage]);
  return null;
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <DataProvider>
          <PageTracker />
          <Routes>
            {/* 封面为独立全屏页：自带暗色左侧栏，不套 Layout（避免浅色 Sidebar 拼黑底） */}
            <Route path="/" element={<CoverPage />} />
            {/* 登录/注册独立全屏，不套 Layout */}
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />
            {/* 公开只读分享页（无需登录，独立全屏） */}
            <Route path="/shared/:id" element={<SharedPage />} />
            <Route element={<Layout />}>
              {/* 智能对话：嵌入侧边栏框架，与「数据上传」共享 DataContext */}
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/clean" element={<CleanPage />} />
              <Route path="/analysis" element={<AnalysisPage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/ethereal-preview" element={<EtherealPreview />} />
              <Route path="/models" element={<AIModelsPage />} />
              <Route path="/reports" element={<ReportsPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              {/* 需登录的页面，套 RequireAuth 守卫 */}
              <Route path="/profile" element={<RequireAuth><ProfilePage /></RequireAuth>} />
              <Route path="/history" element={<RequireAuth><HistoryPage /></RequireAuth>} />
            </Route>
          </Routes>
        </DataProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}
