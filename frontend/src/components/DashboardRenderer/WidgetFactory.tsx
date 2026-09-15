import React, { memo } from 'react';
import type { WidgetSlot, WidgetError } from '../../types/dashboard';
import { KPIWidget } from './WidgetRenderer/KPIWidget';
import { ChartWidget } from './WidgetRenderer/ChartWidget';
import { MapWidget } from './WidgetRenderer/MapWidget';
import { TableWidget } from './WidgetRenderer/TableWidget';
import { InsightWidget } from './WidgetRenderer/InsightWidget';
import { WidgetErrorBoundary } from './WidgetErrorBoundary';

export interface WidgetRendererProps {
  widget: WidgetSlot;
  onFilter?: (field: string, value: string) => void;
  onClick?: (widgetId: string, data: Record<string, unknown>) => void;
  /** 当前高亮标签（用于 Cross Filter / Hover Highlight） */
  highlightLabel?: string | null;
  /** 全局筛选器当前值（field → value），用于筛选高亮 */
  globalFilterValues?: Record<string, string>;
  /** 是否是 Cross Filter 源 */
  isCrossFilterSource?: boolean;
  /** 是否有下钻能力 */
  hasDrillDown?: boolean;
  /** Drill Down 回调 */
  onDrillDown?: (widgetId: string, dimension: string, nextLevel: string) => void;
  /** 错误回调 */
  onWidgetError?: (error: WidgetError) => void;
  /** 尺寸缩放系数（全屏大屏模式传 >1 的值放大图表），默认 1 */
  sizeScale?: number;
}

/**
 * WidgetFactory —— 根据 widget_type 实例化对应组件
 *
 * 采用 Factory Pattern（Map 注册表），不写大量 if-else。
 * 新增 Widget 类型只需在 WIDGET_MAP 中添加一条。
 *
 * 所有 Widget 被 WidgetErrorBoundary 包裹：
 * 单个 Widget 渲染失败 → 显示 Error 占位，不影响 Dashboard。
 */
const WIDGET_MAP: Record<string, React.ComponentType<WidgetRendererProps>> = {
  kpi: KPIWidget,
  chart: ChartWidget,
  map: MapWidget,
  table: TableWidget,
  insight: InsightWidget,
  summary: InsightWidget,
};

export const WidgetFactory: React.FC<WidgetRendererProps> = memo(({ widget, onWidgetError, sizeScale, ...rest }) => {
  const Component = WIDGET_MAP[widget.widget_type] || InsightWidget;

  return (
    <WidgetErrorBoundary widgetId={widget.widget_id} onError={onWidgetError}>
      <Component widget={widget} sizeScale={sizeScale} {...rest} />
    </WidgetErrorBoundary>
  );
});

WidgetFactory.displayName = 'WidgetFactory';
