/**
 * BigScreenCard —— ChatPage 对话流内联数据大屏预览卡片。
 * 消费后端 build_dashboard 工具返回的 tool_result.data.bigscreen：
 *   { widgets: Widget.to_dict()[]（含 id/title/widget_type/chart_config/metadata/importance_score）, widget_count }
 * 复用 DashboardRenderer 的 WidgetFactory 渲染每个 Widget（图表/KPI/表格/洞察）。
 *
 * ★ 排版策略（对话内紧凑版，与 RAG 骨架对齐）：
 *   - header 区（顶部 KPI 条）: grid-cols-2 sm:grid-cols-4，最多 4 个 KPI
 *   - main 区（主要分析 2 栏）: 首图（importance 最高）跨 2 列放大，其余 2 栏
 *   - secondary 区（辅助分析 3 栏）: grid-cols-1 md:grid-cols-2 lg:grid-cols-3
 *   - footer 区（底部表格/洞察）: 单列堆叠
 *   浅色玻璃主题，与 ChatPage 主体风格一致。
 *
 * ★ 全屏模式（仙气风）：
 *   标题栏「⛶ 全屏」按钮 → createPortal 渲染 fixed 全屏层，脱离对话气泡 82% 宽度限制：
 *   - 背景沿用全局仙气粉彩渐变（--bg-1/2/3）
 *   - 顶部毛玻璃标题栏（大屏名 + 组件数 + 实时时间 + 退出）
 *   - 12 列网格：KPI 横条铺满；main 区 importance 最高主图占 8 列大格（sizeScale 放大），
 *     其余 4 列；secondary 4 列；footer 6 列。Esc 或按钮退出。
 */
import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { LayoutDashboard, Maximize2, Minimize2 } from 'lucide-react';
import { WidgetFactory } from './DashboardRenderer/WidgetFactory';

interface ScreenWidget {
  id?: string;
  widget_id?: string;
  title?: string;
  widget_type?: string;
  chart_config?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  importance_score?: number;
  display_role?: string;
  preferred_size?: string;
}

type SlotZone = 'header' | 'main' | 'secondary' | 'footer';

const ZONE_TITLES: Record<SlotZone, string> = {
  header: '核心指标',
  main: '主要分析',
  secondary: '辅助分析',
  footer: '补充信息',
};

/** 仙气粉彩渐变（与 index.css body 背景同源，全屏层铺满视口） */
const ETHEREAL_BG = 'linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 45%, var(--bg-3) 100%)';

const BigScreenCard: React.FC<{ bigscreen: { widgets?: ScreenWidget[]; widget_count?: number } }> = ({
  bigscreen,
}) => {
  const widgets = bigscreen.widgets || [];
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [now, setNow] = useState(() => new Date());

  // 统一 widget_id（Widget.to_dict 输出 id），并把 widget 投到 slot_zone 分组
  const slots = widgets.map((w) => ({
    ...w,
    widget_id: w.widget_id || w.id,
    _zone: ((w.metadata?.slot_zone as SlotZone) || 'secondary') as SlotZone,
  }));

  const groups: Record<SlotZone, typeof slots> = {
    header: [],
    main: [],
    secondary: [],
    footer: [],
  };
  for (const s of slots) {
    if (groups[s._zone]) groups[s._zone].push(s);
    else groups.secondary.push(s);
  }

  // main 区按 importance 降序：首位即"主图"（对话内跨 2 列 / 全屏占 8 列大格）
  const mainRanked = useMemo(
    () => [...groups.main].sort((a, b) => (b.importance_score ?? 0) - (a.importance_score ?? 0)),
    [groups.main],
  );

  const hasAny = slots.length > 0;
  if (!hasAny) return null;

  // ─── 全屏模式副作用：Esc 退出 + 锁 body 滚动 + 每分钟刷新时间 ───
  useEffect(() => {
    if (!isFullscreen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsFullscreen(false);
    };
    window.addEventListener('keydown', onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const timer = setInterval(() => setNow(new Date()), 30_000);
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = prevOverflow;
      clearInterval(timer);
    };
  }, [isFullscreen]);

  const timeLabel = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(
    now.getDate(),
  ).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;

  // ─── 全屏层（portal 到 body，脱离对话容器宽度限制） ───
  const fullscreenLayer = isFullscreen
    ? createPortal(
        <div
          className="fixed inset-0 z-50 flex flex-col"
          style={{ background: ETHEREAL_BG }}
          role="dialog"
          aria-label="数据大屏全屏视图"
        >
          {/* 顶部毛玻璃标题栏 */}
          <div className="flex items-center gap-3 px-6 py-3 bg-white/60 backdrop-blur-md border-b border-white/70 shadow-sm">
            <LayoutDashboard className="w-5 h-5 text-sky-600" />
            <h2 className="text-base font-semibold text-slate-800">数据大屏</h2>
            <span className="text-xs px-2 py-0.5 rounded-full bg-violet-500/15 text-violet-700 border border-violet-400/40">
              {bigscreen.widget_count ?? widgets.length} 个组件
            </span>
            <span className="ml-auto text-sm text-slate-600 tabular-nums">{timeLabel}</span>
            <button
              onClick={() => setIsFullscreen(false)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm font-medium
                         bg-white/70 border border-slate-300/60 text-slate-700
                         hover:bg-white hover:border-sky-300 transition-colors"
            >
              <Minimize2 className="w-4 h-4" /> 退出全屏
            </button>
          </div>

          {/* 主体：12 列网格，可滚动 */}
          <div className="flex-1 overflow-y-auto p-6">
            {/* KPI 横条：数量自适应铺满 */}
            {groups.header.length > 0 && (
              <div
                className="grid gap-4 mb-5"
                style={{
                  gridTemplateColumns: `repeat(${Math.min(Math.max(groups.header.length, 2), 5)}, minmax(0, 1fr))`,
                }}
              >
                {groups.header.map((slot, i) => (
                  <div
                    key={slot.widget_id || i}
                    className="p-4 min-h-[110px] flex flex-col justify-between"
                  >
                    <div className="text-sm font-medium text-slate-500 truncate">
                      {slot.title || 'KPI'}
                    </div>
                    <WidgetFactory widget={slot as never} sizeScale={1.5} />
                  </div>
                ))}
              </div>
            )}

            {/* 12 列主网格：主图 8 列 / 辅图 4 列 */}
            <div className="grid grid-cols-12 gap-4">
              {mainRanked.map((slot, i) => (
                <div
                  key={slot.widget_id || i}
                  className={
                    i === 0
                      ? 'col-span-12 lg:col-span-8'
                      : 'col-span-12 md:col-span-6 lg:col-span-4'
                  }
                >
                  <WidgetFactory widget={slot as never} sizeScale={i === 0 ? 1.8 : 1.4} />
                </div>
              ))}
              {groups.secondary.map((slot, i) => (
                <div key={slot.widget_id || i} className="col-span-12 md:col-span-6 lg:col-span-4">
                  <WidgetFactory widget={slot as never} sizeScale={1.4} />
                </div>
              ))}
              {groups.footer.map((slot, i) => (
                <div key={slot.widget_id || i} className="col-span-12 lg:col-span-6">
                  <WidgetFactory widget={slot as never} sizeScale={1.3} />
                </div>
              ))}
            </div>
          </div>
        </div>,
        document.body,
      )
    : null;

  // ─── 对话内紧凑版 ───
  return (
    <div className="mt-3 rounded-2xl border border-slate-200/80 bg-white/80 backdrop-blur-md shadow-[0_8px_30px_rgba(56,189,248,0.12)] overflow-hidden">
      {/* 标题栏 —— 浅色玻璃风 */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-200/70 bg-gradient-to-r from-sky-50 to-violet-50">
        <LayoutDashboard className="w-4 h-4 text-sky-600" />
        <span className="text-sm font-semibold text-slate-800">数据大屏预览</span>
        <span className="ml-1 text-[11px] text-slate-500">
          {bigscreen.widget_count ?? widgets.length} 个组件
        </span>
        {/* 全屏入口：脱离对话气泡宽度限制，获得真正的大屏体验 */}
        <button
          onClick={() => setIsFullscreen(true)}
          className="ml-auto inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium
                     bg-white/70 border border-slate-300/60 text-slate-700
                     hover:bg-white hover:border-sky-300 hover:text-sky-700 transition-colors"
        >
          <Maximize2 className="w-3.5 h-3.5" /> 全屏查看
        </button>
      </div>

      <div className="p-4 space-y-4">
        {/* ★ 顶部 KPI 条 */}
        {groups.header.length > 0 && (
          <section
            aria-label={ZONE_TITLES.header}
            className="grid grid-cols-2 sm:grid-cols-4 gap-3"
          >
            {groups.header.map((slot, i) => (
              <KpiShell key={slot.widget_id || i} slot={slot} />
            ))}
          </section>
        )}

        {/* ★ 主要分析：首图（importance 最高）跨 2 列放大，突出视觉重心 */}
        {mainRanked.length > 0 && (
          <section aria-label={ZONE_TITLES.main}>
            <ZoneHeader title={ZONE_TITLES.main} count={mainRanked.length} />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              {mainRanked.map((slot, i) => (
                <div key={slot.widget_id || i} className={i === 0 ? 'lg:col-span-2' : ''}>
                  <WidgetShell slot={slot} sizeScale={i === 0 ? 1.35 : 1} />
                </div>
              ))}
            </div>
          </section>
        )}

        {/* ★ 辅助分析 3 栏 */}
        {groups.secondary.length > 0 && (
          <section aria-label={ZONE_TITLES.secondary}>
            <ZoneHeader title={ZONE_TITLES.secondary} count={groups.secondary.length} />
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {groups.secondary.map((slot, i) => (
                <WidgetShell key={slot.widget_id || i} slot={slot} />
              ))}
            </div>
          </section>
        )}

        {/* ★ 底部表格/洞察 */}
        {groups.footer.length > 0 && (
          <section aria-label={ZONE_TITLES.footer}>
            <ZoneHeader title={ZONE_TITLES.footer} count={groups.footer.length} />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              {groups.footer.map((slot, i) => (
                <WidgetShell key={slot.widget_id || i} slot={slot} />
              ))}
            </div>
          </section>
        )}
      </div>

      {fullscreenLayer}
    </div>
  );
};

const ZoneHeader: React.FC<{ title: string; count: number }> = ({ title, count }) => (
  <div className="flex items-center gap-2 mb-2">
    <span className="inline-block w-1 h-3.5 rounded-sm bg-gradient-to-b from-sky-400 to-violet-400" />
    <span className="text-[12px] font-semibold text-slate-700">{title}</span>
    <span className="text-[11px] text-slate-400">· {count}</span>
  </div>
);

/**
 * KpiShell —— 顶部 KPI 条专用：紧凑卡片样式，强调数值
 */
const KpiShell: React.FC<{ slot: ScreenWidget & { widget_id: string } }> = ({ slot }) => (
  <div className="p-3 min-h-[88px] flex flex-col justify-between">
    <div className="text-[11px] font-medium text-slate-500 truncate">
      {slot.title || 'KPI'}
    </div>
    <WidgetFactory widget={slot as never} />
  </div>
);

/**
 * WidgetShell —— 普通图表/表格/洞察的容器（sizeScale 供主图放大）
 */
const WidgetShell: React.FC<{ slot: ScreenWidget & { widget_id: string }; sizeScale?: number }> = ({
  slot,
  sizeScale,
}) => (
  <div className="h-full">
    <WidgetFactory widget={slot as never} sizeScale={sizeScale} />
  </div>
);

export default BigScreenCard;
