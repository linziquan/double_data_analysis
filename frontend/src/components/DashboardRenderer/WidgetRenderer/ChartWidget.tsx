import React, { memo, useRef, useEffect, useMemo } from 'react';
import type { WidgetSlot } from '../../../types/dashboard';
import { useDashboardTheme } from '../ThemeProvider';
import { useWidgetAnimation } from '../hooks';
import { EtherealChart } from '../../EtherealCharts/EtherealChart';
import { EtherealBubbleChart } from '../../EtherealCharts/EtherealBubbleChart';
import { EtherealFunnelChart } from '../../EtherealCharts/EtherealFunnelChart';

interface ChartWidgetProps {
  widget: WidgetSlot;
  onFilter?: (field: string, value: string) => void;
  onClick?: (widgetId: string, data: Record<string, unknown>) => void;
  highlightLabel?: string | null;
  globalFilterValues?: Record<string, string>;
  isCrossFilterSource?: boolean;
  hasDrillDown?: boolean;
  onDrillDown?: (widgetId: string, dimension: string, nextLevel: string) => void;
  /** 尺寸缩放系数（全屏大屏模式传 >1 的值放大图表高度），默认 1 */
  sizeScale?: number;
}

/**
 * ChartWidget —— 通用图表渲染组件
 *
 * 设计原则：项目里所有图表都走仙气粉彩组件库（Ethereal*Chart），
 * 后端只负责决定 chart_type / data / option，渲染层统一靠 EtherealChart 派发，
 * 不再落到裸 ECharts（参考 EtherealChart.tsx 第 195 行：fallback 路径会丢仙气渐变）。
 */
export const ChartWidget: React.FC<ChartWidgetProps> = memo(({ widget, onClick, hasDrillDown, onDrillDown, sizeScale = 1 }) => {
  const theme = useDashboardTheme();

  // Animation
  const { ref: animRef, animationClass, animationStyle } = useWidgetAnimation({
    type: 'scale-in',
    delay: (widget.importance_score % 5) * 50,
  });

  // 后端送来的 chart_config.option 是 ECharts option（含 xAxis/yAxis/series/...）
  const chartNode = (widget.chart_config?.option as Record<string, unknown>) || {};
  // chartNode/data 是 Ethereal* 组件的统一入参
  const data = (widget.chart_config?.data as Array<Record<string, unknown>> | undefined) || undefined;
  // chart_type：未识别时回落到 bar（避免 EtherealChart 落到 default 分支走裸 ECharts）
  const chartType =
    widget.chart_type ||
    (chartNode.series?.[0]?.type as string | undefined) ||
    'bar';
  // slot 主要用于多档槽位的显示分支；这里用 widget_id 兜底让 EtherealChart 命中
  const slot = widget.widget_id || 'chart';

  // 高度：与 widget.preferred_size 对齐，与 Ethereal 子组件默认值一致
  // sizeScale > 1（全屏大屏模式）时按比例放大，摆脱对话内 220px 小图
  const height = useMemo<number | string>(() => {
    let base: number;
    switch (widget.preferred_size) {
      case 'HERO':
        base = 360;
        break;
      case 'LARGE':
        base = 300;
        break;
      case 'MEDIUM':
        base = 260;
        break;
      case 'SMALL':
      default:
        base = 220;
        break;
    }
    return Math.round(base * sizeScale);
  }, [widget.preferred_size, sizeScale]);

  // Drill Down
  const handleDrillDownClick = () => {
    const ddInfo = widget.metadata?.drill_down as { dimension: string; next_level: string } | undefined;
    if (ddInfo) onDrillDown?.(widget.widget_id, ddInfo.dimension, ddInfo.next_level);
  };

  const isBubbleScatter = chartType === 'bubble' || chartType === 'scatter';
  const isFunnel = chartType === 'funnel';

  // 组件不可用时占位（兼容方案，理论上后端必给 option）
  if (!chartNode || Object.keys(chartNode).length === 0) {
    return (
      <div ref={animRef}
        className={`relative p-4 ${animationClass}`}
        style={animationStyle}
        onClick={() => onClick?.(widget.widget_id, {})}
      >
        <div className="flex items-center justify-center h-full text-slate-600 text-xs">
          暂无图表数据 · {widget.title}
        </div>
      </div>
    );
  }

  return (
    <div ref={animRef}
      className={`relative ${animationClass}`}
      style={{
        padding: theme.cardPadding,
        ...animationStyle,
      }}
      onClick={() => onClick?.(widget.widget_id, {})}
    >
      {hasDrillDown && (
        <button onClick={(e) => { e.stopPropagation(); handleDrillDownClick(); }}
          className="absolute top-2 right-2 text-xs px-1.5 py-0.5 rounded
            bg-[var(--db-accent)]/10 text-[var(--db-accent)] border border-[var(--db-accent)]/20
            hover:bg-[var(--db-accent)]/20 transition-colors z-10"
          title="下钻分析"
        >
          ↓ 下钻
        </button>
      )}

      {isBubbleScatter ? (
        <EtherealBubbleChart chartNode={chartNode} height={height} />
      ) : isFunnel ? (
        <EtherealFunnelChart
          chartNode={chartNode}
          title={widget.title}
          height={height}
          cardBgUrl=""
        />
      ) : (
        <EtherealChart
          slot={slot}
          chartType={chartType}
          chartNode={chartNode}
          data={data}
          title={widget.title}
          height={height}
        />
      )}
      {/* 图表下方不再渲染 description 文字说明（用户要求大屏只有图表，
          文字堆在图下会让大屏看起来像图表列表的堆砌） */}
    </div>
  );
});

ChartWidget.displayName = 'ChartWidget';
