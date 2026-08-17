/**
 * BigScreenCard —— ChatPage 对话流内联数据大屏预览卡片。
 * 消费后端 build_dashboard 工具返回的 tool_result.data.bigscreen：
 *   { widgets: Widget.to_dict()[]（含 id/title/widget_type/chart_config/metadata/importance_score/preferred_size）,
 *     widget_count, layout: { blocks: [{block_id,title,widget_ids,relation_type}], order: [id...] } }
 * 复用 DashboardRenderer 的 WidgetFactory 渲染每个 Widget（图表/KPI/表格/洞察）。
 *
 * 叙事编排：优先消费 layout 做非对称灵动网格（依 preferred_size 映射 col-span），
 * 每个卡片左上角显示关联标签（递进/对照/相关）。无 layout 时降级为原均分网格（向后兼容）。
 */
import React from 'react';
import { LayoutDashboard } from 'lucide-react';
import { WidgetFactory } from './DashboardRenderer/WidgetFactory';

interface ScreenWidget {
  id?: string;
  widget_id?: string;
  title?: string;
  widget_type?: string;
  chart_config?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  importance_score?: number;
  // 后端编排写回的字段（硬伤1：必须补类型，否则 TS 报属性不存在）
  preferred_size?: string;
  narrative?: {
    block_id?: string;
    block_title?: string;
    relation_type?: string;
  };
}

interface BigScreenLayout {
  blocks?: Array<{
    block_id?: string;
    title?: string;
    widget_ids?: string[];
    relation_type?: string;
  }>;
  order?: string[];
}

interface BigScreenData {
  widgets?: ScreenWidget[];
  widget_count?: number;
  layout?: BigScreenLayout;
}

const RELATION_LABEL: Record<string, string> = {
  progressive: '递进',
  contrast: '对照',
  related: '相关',
  solo: '独立',
};

// 仙气粉彩：关联标签配色（递进=青、对照=粉、相关=金、独立=灰）
const RELATION_STYLE: Record<string, string> = {
  progressive: 'bg-cyan-400/15 text-cyan-200 border-cyan-300/30',
  contrast: 'bg-pink-400/15 text-pink-200 border-pink-300/30',
  related: 'bg-amber-400/15 text-amber-200 border-amber-300/30',
  solo: 'bg-white/10 text-slate-300 border-white/20',
};

// 由 preferred_size 映射 col-span（仅 hero/large/medium/small 四值，禁 sidebar）
function sizeToColSpan(size?: string): string {
  switch (size) {
    case 'hero':
      return 'sm:col-span-2';
    case 'large':
    case 'medium':
      return 'sm:col-span-1';
    case 'small':
      return 'sm:col-span-1';
    default:
      return 'sm:col-span-1';
  }
}

const BigScreenCard: React.FC<{ bigscreen: BigScreenData }> = ({ bigscreen }) => {
  const widgets = bigscreen.widgets || [];
  const layout = bigscreen.layout;

  // 按 layout.order 重排（无 layout 时保持原顺序）
  const ordered = React.useMemo(() => {
    if (layout?.order && layout.order.length) {
      const byId = new Map(widgets.map((w) => [w.id || w.widget_id || '', w]));
      const orderedWidgets: ScreenWidget[] = [];
      for (const oid of layout.order) {
        const w = byId.get(oid);
        if (w) orderedWidgets.push(w);
      }
      // 兜底：order 未覆盖的 widget 追加在末尾
      for (const w of widgets) {
        const key = w.id || w.widget_id || '';
        if (!layout.order.includes(key)) orderedWidgets.push(w);
      }
      return orderedWidgets;
    }
    return widgets;
  }, [widgets, layout]);

  // WidgetFactory 需要 widget_id 字段（Widget.to_dict 输出 id），做一次映射
  const slots = ordered.map((w) => ({ ...w, widget_id: w.widget_id || w.id }));

  const hasLayout = Boolean(layout && layout.blocks && layout.blocks.length);

  return (
    <div className="mt-2 rounded-2xl border border-white/10 bg-gradient-to-br from-[#0F172A] to-[#020617] shadow-[0_8px_30px_rgba(56,189,248,0.18)] overflow-hidden">
      {/* 标题栏 */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-white/10 bg-gradient-to-r from-violet-500/10 to-sky-400/5">
        <LayoutDashboard className="w-4 h-4 text-sky-300" />
        <span className="text-sm font-semibold text-slate-100">数据大屏预览</span>
        <span className="ml-auto text-[11px] text-slate-400">
          {bigscreen.widget_count ?? widgets.length} 个组件
        </span>
      </div>

      {/* Widget 网格：有 layout 走非对称灵动网格，否则降级原均分网格 */}
      <div className={hasLayout ? 'p-3 grid grid-cols-1 sm:grid-cols-2 gap-3' : 'p-3 grid grid-cols-1 sm:grid-cols-2 gap-3'}>
        {slots.map((slot, i) => {
          const relation = slot.narrative?.relation_type ?? 'solo';
          const label = RELATION_LABEL[relation] ?? '独立';
          const tagStyle = RELATION_STYLE[relation] ?? RELATION_STYLE.solo;
          const colSpan = hasLayout ? sizeToColSpan(slot.preferred_size) : 'sm:col-span-1';
          return (
            <div
              key={slot.widget_id || i}
              className={`rounded-xl border border-white/10 bg-white/[0.03] backdrop-blur-sm overflow-hidden relative
                          hover:border-violet-400/40 hover:shadow-[0_0_18px_rgba(139,92,246,0.25)] transition-all ${colSpan}`}
            >
              {/* 关联标签（仙气粉彩小胶囊） */}
              {relation !== 'solo' && (
                <span
                  className={`absolute top-2 left-2 z-10 px-2 py-0.5 rounded-full text-[10px] font-medium border ${tagStyle} backdrop-blur-sm`}
                >
                  {label}
                </span>
              )}
              <WidgetFactory widget={slot as never} />
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default BigScreenCard;
