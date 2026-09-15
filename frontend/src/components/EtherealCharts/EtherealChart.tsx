/**
 * 仙气图表统一分发组件（React 版）
 * 对应原版「可视化模板库/utils.js → renderChartBySlot」的 CHART_REGISTRY 分发逻辑
 *
 * 用法：<EtherealChart slot="rfm_pie" chartNode={{...}} height={360} />
 *
 * slot → 组件的映射照搬原版 CHART_REGISTRY，未改动语义。
 */
import React from 'react';
import { EtherealPieChart } from './EtherealPieChart';
import { EtherealLineChart } from './EtherealLineChart';
import { EtherealBarChart } from './EtherealBarChart';
import { EtherealRadarChart } from './EtherealRadarChart';
import { EtherealBubbleChart } from './EtherealBubbleChart';
import { EtherealNetworkChart } from './EtherealNetworkChart';
import { EtherealRankChart } from './EtherealRankChart';
import { EtherealDimOffsetChart } from './EtherealDimOffsetChart';
import { EtherealRetentionMatrix } from './EtherealRetentionMatrix';
import { EtherealMetricCard } from './EtherealMetricCard';
import { EtherealTable } from './EtherealTable';
import { EtherealDualAxisChart } from './EtherealDualAxisChart';
import { EtherealFunnelChart } from './EtherealFunnelChart';
import EChartView from '../EChartView';

interface Props {
  slot: string;
  /** 图表类型（与后端 chart_type 对齐），如 pie/bar/line/radar/heatmap... 按类型分发，不依赖具体 slot 名 */
  chartType?: string;
  /** 从分析包 / mock JSON 提取的 chart 节点（含 data / series / columns 等） */
  chartNode?: Record<string, unknown>;
  /** 扁平数据（部分组件可直接传） */
  data?: Array<Record<string, unknown>>;
  title?: string;
  filter?: Record<string, string>;
  cardBgUrl?: string;
  /** 扇区染色纹理图（饼图专用），传 base64 时覆盖 UMD 内被 stub 的占位图 */
  sliceTextureUrl?: string;
  /** 下三角矩阵专用：'percent'（留存率，默认）| 'number'（客单价/净毛利等数值） */
  valueFormat?: 'percent' | 'number';
  /** 调用方显式指定图表高度（px 或 '100%' 等）；不传则回落到 '100%' 父容器自适应 */
  height?: number | string;
}

/** 已实现的仙气组件类型 */
const IMPLEMENTED = new Set(['pie', 'bar', 'line', 'radar', 'dual_axis', 'heatmap', 'hbar', 'bubble', 'ranking', 'table', 'metric', 'graph', 'funnel']);

/** 把后端五花八门的 chart_type 归一化为有限的仙气组件类型；
 * 未实现的类型（funnel/ranking/sankey/graph 等）原样返回，由调用方显式报错，不静默兜底。 */
function normalizeChartType(t: string): string {
  const s = (t || '').toLowerCase();
  if (s === 'pie' || s === 'ring' || s === 'pie_chart' || s === 'donut') return 'pie';
  if (s === 'bar' || s.startsWith('bar_') || s.endsWith('_bar') || s === 'vertical_bar' || s === 'v_bar') return 'bar';
  if (
    s === 'line' ||
    s.startsWith('line_') ||
    s.endsWith('_line') ||
    s.endsWith('_trend') ||
    s === 'rfm_line' ||
    s === 'area' ||
    s === 'area_chart'
  ) return 'line';
  if (s === 'radar') return 'radar';
  if (s === 'dual_axis' || s === 'dual' || s === 'dualbar_line' || s === 'dual_axis_chart') return 'dual_axis';
  if (s === 'heatmap' || s === 'cohort_heatmap') return 'heatmap';
  if (s === 'bubble_matrix' || s === 'bubble') return 'bubble';
  if (s === 'graph' || s === 'chord' || s === 'chord_diagram') return 'graph';
  if (s === 'hbar_family' || s === 'dim_offset' || s === 'dim2_offset') return 'hbar';
  if (s === 'ranking' || s === 'horizontal_bar' || s === 'h_bar' || s === 'hbar_rank') return 'ranking';
  if (s === 'table' || s === 'cohort_table' || s === 'rank_table' || s === 'analysis_table') return 'table';
  if (s === 'metric' || s === 'kpi' || s === 'card' || s === 'metric_card') return 'metric';
  if (s === 'funnel' || s === 'funnel_chart') return 'funnel';
  // 未实现的类型（sankey/waterfall/word_cloud）原样返回走报错分支
  return s;
}

/**
 * 下三角矩阵（cohort_heatmap）的 valueFormat 自动判定：
 * - 留存率（cohort_retention）是 0~1 比例 → 默认 percent（×100 + %）
 * - 客单价（cohort_arpu）/ 净毛利（cohort_c_netmargin_heat）是金额 → 强制 number（纯数值不带 %）
 * 调用方若显式传 valueFormat，以调用方为准（兼容手动覆盖）。
 */
function resolveValueFormat(
  chartType: string | undefined,
  slot: string,
  explicit?: 'percent' | 'number',
): 'percent' | 'number' {
  if (explicit) return explicit;
  if (chartType === 'cohort_heatmap' && slot !== 'cohort_retention') return 'number';
  return 'percent';
}

export const EtherealChart: React.FC<Props> = ({ slot, chartType, chartNode, data, title, filter, cardBgUrl, sliceTextureUrl, valueFormat, height }) => {
  const raw = data as Array<Record<string, unknown>> | undefined;

  // 优先用 chartType；其次从 chartNode 里取；都没有再用 slot 兜底（兼容老预览页写法）
  const rawType =
    chartType ||
    (chartNode?.chart_type as string) ||
    slot; // slot 本身有时也含类型线索（如 rfm_pie / cohort_a_line）
  let type = normalizeChartType(rawType);

  // ★ 设计原则：完全信任后端下发的 chart_type。柱状图(bar) 与 排行图(ranking) 由后端
  //   在生成时就已经区分好（ranking 用于 TopN 排序展示，bar 用于普通柱状图），前端不再
  //   通过 slot 命名或数据形态二次猜测，避免把正常的柱状图误判成排行图 / 反之。

  // ★ 调试钩子：默认关闭（生产），打开方式：浏览器 console 执行
  //   window.__datamind_chart_debug__ = true; 刷新后所有归一结果会打印到 console
  if (typeof window !== 'undefined' && (window as Record<string, unknown>).__datamind_chart_debug__) {
    // eslint-disable-next-line no-console
    console.log('[EC]', { rawType, slot, finalType: type, dataSample: (Array.isArray(data) && data[0]) || null });
    // ★ 额外：把排名图被错误路径渲染的关键日志也打到 console，方便定位
    if (rawType === 'bar' && /客户生命周期价值|客渠道|商品类目|流量来源/.test(String(slot || '') + (chartNode?.title as string || ''))) {
      // eslint-disable-next-line no-console
      console.warn('[EC] CLV图路由异常', { rawType, slot, finalType: type, chartType: rawType, chartNode });
    }
  }

  // ★ 调用方不传 height 时回落 undefined，让各子组件使用自身默认高度（如饼图 360、柱状图 360），
  //   不再强制 '100%' 顶掉子组件默认值导致容器高度塌陷为 0（饼图空白根因）。显式传 height 时仍尊重调用方。
  const wrapperHeight: number | string | undefined = height ?? undefined;
  switch (type) {
    case 'pie':
      return <EtherealPieChart option={chartNode as Record<string, unknown>} title={title} cardBgUrl={cardBgUrl} sliceTextureUrl={sliceTextureUrl} height={wrapperHeight} />;
    case 'heatmap': {
      // 两种数据来源，最终都用仙气矩阵组件渲染（不回退老组件）：
      // 1) 分析包路径：扁平清单在 chartNode.data 或 raw(=chart.raw_data)；
      // 2) 仪表盘路径：后端 cohort 数据在 option.series[0].data，无扁平清单 → 从 series 反推。
      let matrixData = (chartNode?.data as Array<Record<string, unknown>>) || raw || [];
      if (matrixData.length === 0 && chartNode?.series?.[0]?.data) {
        // 从 ECharts heatmap option 反推仙气矩阵要的扁平清单 [{首单月, Index_j, value}]
        const seriesData = (chartNode.series[0].data as Array<[number, number, number]>);
        const xAxisData = (chartNode.xAxis?.data as Array<string>) || [];
        const yAxisData = (chartNode.yAxis?.data as Array<string>) || [];
        matrixData = seriesData.map(([xi, yi, v]) => ({
          首单月: yAxisData[yi] ?? String(yi),
          Index_j: xi,
          value: v,
        }));
      }
      return <EtherealRetentionMatrix chartNode={chartNode} rawData={matrixData} title={title} cardBgUrl={cardBgUrl} valueFormat={resolveValueFormat(rawType, slot, valueFormat)} height={wrapperHeight} />;
    }
    case 'bar':
      return <EtherealBarChart chartNode={chartNode} title={title} data={data as Array<Record<string, unknown>> | undefined} cardBgUrl={cardBgUrl} height={wrapperHeight} />;
    case 'line':
      return <EtherealLineChart chartNode={chartNode} title={title} cardBgUrl={cardBgUrl} height={wrapperHeight} />;
    case 'metric':
      return <EtherealMetricCard metricData={chartNode as { title?: string; label?: string; value?: number | string; change?: number | string; unit?: string }} />;
    case 'dual_axis':
      return <EtherealDualAxisChart chartNode={chartNode} title={title} cardBgUrl={cardBgUrl} height={wrapperHeight} />;
    case 'radar':
      return <EtherealRadarChart chartNode={chartNode} title={title} cardBgUrl={cardBgUrl} height={wrapperHeight} />;
    case 'table':
      return <EtherealTable chartNode={chartNode} title={title} cardBgUrl={cardBgUrl} />;
    case 'hbar': {
      // 维度偏移图：优先用 data prop（后端 ChartData.data 扁平清单，含 维度/维度取值/偏移值），
      // 与预览页 VisualizationRenderer 路径一致；不回退老组件、不改 EtherealDimOffsetChart 组件内部。
      const hbarData = (data && data.length > 0)
        ? data
        : (chartNode?.data as Array<Record<string, unknown>>) || [];
      return <EtherealDimOffsetChart chartNode={{ ...chartNode, data: hbarData }} title={title} filter={filter} cardBgUrl={cardBgUrl} height={wrapperHeight} />;
    }
    case 'ranking': {
      const rankData = (data && data.length > 0)
        ? data
        : (chartNode?.data as Array<Record<string, unknown>>) || [];
      return <EtherealRankChart chartNode={{ ...chartNode, data: rankData }} title={title} cardBgUrl={cardBgUrl} height={wrapperHeight} />;
    }
    case 'bubble': {
      let bubbleData = (raw && raw.length > 0)
        ? raw
        : ((chartNode?.data as Array<Record<string, unknown>>) || []);
      if (bubbleData.length === 0 && (chartNode as { series?: Array<{ data?: Array<{ value?: [unknown, unknown, unknown] }> }> })?.series?.[0]?.data) {
        const seriesData = (chartNode as { series: Array<{ data: Array<{ value?: [unknown, unknown, unknown] }> }> }).series[0].data;
        bubbleData = seriesData.map((pt) => {
          const v = pt.value || [null, null, null];
          return {
            标签: v[0] ?? '',
            聚类: v[1] ?? '',
            人数: v[2] ?? 0,
          };
        });
      }
      return <EtherealBubbleChart chartNode={{ ...chartNode, data: bubbleData }} title={title} cardBgUrl={cardBgUrl} height={wrapperHeight} />;
    }
    case 'graph': {
      const netData = (raw && raw.length > 0)
        ? raw
        : ((chartNode?.data as Array<Record<string, unknown>>) || []);
      return <EtherealNetworkChart chartNode={{ ...chartNode, data: netData }} title={title} cardBgUrl={cardBgUrl} height={wrapperHeight} />;
    }
    case 'funnel': {
      return <EtherealFunnelChart chartNode={chartNode} title={title} cardBgUrl={cardBgUrl} height={wrapperHeight} />;
    }
    default: {
      // 暂无量身定制的仙气组件，直接用原生 ECharts 渲染（保证图表永远可见，不退化成占位）
      // option 不合法时（无 series / xAxis/yAxis 缺失）静默回退为占位卡片，避免污染其他图表渲染
      const opt = chartNode as Record<string, unknown> | undefined;
      const hasSeries = Array.isArray(opt?.series) && (opt!.series as unknown[]).length > 0;
      if (!hasSeries) {
        return (
          <div className="flex items-center justify-center w-full h-full text-slate-600 text-xs">
            暂不支持的图表类型：{rawType}
          </div>
        );
      }
      return <EChartView
        option={opt as never}
        title={title}
        hideTitle
        height={typeof wrapperHeight === 'number' ? wrapperHeight : 400}
      />;
    }
  }
};

export default EtherealChart;
