/* DataMind AI - API 响应类型 */

export interface ApiResponse<T = unknown> {
  success: boolean;
  [key: string]: T | boolean | string | number | undefined;
}

export interface UploadResponse {
  session_id: string;
  success: boolean;
  file_name: string;
  rows: number;
  columns: number;
  memory_usage: string;
  total_missing: number;
  duplicate_rows: number;
  preview: Record<string, unknown>[];
  column_info: {
    name: string;
    dtype: string;
    missing: number;
    missing_rate: number;
    unique: number;
    sample: string;
  }[];
  dataset_id: string;
  used_bytes: number;
  quota_bytes: number;
  file_size_bytes: number;
  /** 顶层列名数组（首表），供单表兜底路径使用 */
  column_names?: string[];
  /** 多 sheet Excel 时返回所有被识别出的数据表清单；单表时长度为 1 */
  datasets?: UploadDatasetItem[];
  sheet_count?: number;
}

/** 上传响应中单个数据表的元信息（多 sheet 拆分后每个 sheet 一项） */
export interface UploadDatasetItem {
  dataset_id: string;
  file_name: string;
  rows: number;
  columns: number;
  memory_usage: string;
  total_missing: number;
  duplicate_rows: number;
  preview: Record<string, unknown>[];
  column_info: {
    name: string;
    dtype: string;
    missing: number;
    missing_rate: number;
    unique: number;
    sample: string;
  }[];
  /** 该数据表的列名数组（与 DatasetInfo.columns 语义一致） */
  column_names?: string[];
}

export interface PreviewResponse {
  success: boolean;
  preview: Record<string, unknown>[];
  total_rows: number;
}

export interface StatsResponse {
  success: boolean;
  stats: Record<string, unknown>[] | Record<string, unknown>;
  columns: string[];
}

export interface ChartResponse {
  success: boolean;
  figure: PlotlyFigure;
}

export interface PlotlyFigure {
  data: Record<string, unknown>[];
  layout: Record<string, unknown>;
}

/** ECharts 图表响应 */
export interface EChartResponse {
  success: boolean;
  option: Record<string, unknown>;
}

/** ECharts 仪表盘图表项 */
export interface EChartItem {
  title: string;
  option: Record<string, unknown>;
  /** 图表类型：''=普通图表, 'table'=同环比表格 */
  chart_type?: string;
  /** 同环比表格数据 */
  table_data?: Record<string, unknown>;
  /** 原始扁平 rows（ChartData.data），同期群等吃扁平清单的组件使用 */
  raw_data?: Record<string, unknown>[];
}

export interface InsightsResponse {
  success: boolean;
  insights: string;
}

export interface ChatResponse {
  success: boolean;
  answer: string;
  intents?: Array<Record<string, string>>;
}

/** 智能体聊天响应（function calling 结构化输出） */
export interface ChatSendResponse {
  success: boolean;
  kind: 'text' | 'choice' | 'tool_executing';
  content: string;
  choices: Array<{ id: string; label: string; description?: string }>;
  tool_results: Array<{ tool: string; status: string; summary?: string; data?: any }>;
  data_preview?: {
    rows?: number;
    columns?: string[];
    head?: Array<Record<string, any>>;
  } | null;
}

/** 报告 section（五阶段分析流水线输出） */
export interface ReportSection {
  type: 'overview' | 'kpi' | 'trend' | 'structure' | 'top' | 'anomaly' | 'conclusion' | 'suggestions' | 'next_steps' | 'error';
  title: string;
  /** 连续文字分析（Markdown，章节内禁用标题，层级由 type/title 决定） */
  content?: string;
  /** 本章节正文引用的图表标题数组（由后端映射到 section_charts） */
  chart_titles?: string[];
  /** 后端把 chart_titles 解析成的具体图表（含 option/raw_data），前端就近插图 */
  section_charts?: PackageChartItem[];
  /** 保留：旧 insights 结构（fallback 或极旧后端可能返回） */
  insights?: Array<string | ReportInsight>;
  /** next_steps section 专有字段 */
  charts_to_create?: ChartToCreate[];
  action_items?: ActionItem[];
}

/** 单条洞察（对象格式，包含规则映射） */
export interface ReportInsight {
  chart_title: string | null;
  chart_type: string | null;
  table_type: string | null;
  rule_id: string | null;
  insight_label: string | null;
  analysis: string;
}

/** next_steps：推荐生成的图表 */
export interface ChartToCreate {
  chart_title: string;
  chart_type: string;
  rule_id: string;
  x_axis: string;
  y_axis: string;
  value: string;
  guide: string;
}

/** next_steps：操作清单项 */
export interface ActionItem {
  priority: number;
  action: string;
}

/** 报告降级说明：对外归因 AI 接口，不泄露原始堆栈 */
export interface ReportDegradation {
  degraded: boolean;
  reason?: 'llm_timeout' | 'llm_unavailable' | 'llm_error';
  message?: string;
  canRegenerate?: boolean;
}

export interface AIReportResponse {
  success: boolean;
  sections: ReportSection[];
  /** LLM 根据数据生成的业务标题，为空时前端用默认文案 */
  report_title?: string;
  /** 报告引用的可视化图表（ECharts option），供 ReportsPage 内联渲染 */
  charts?: PackageChartItem[];
  raw_analysis?: Record<string, unknown>;
  warning?: string;
  degradation?: ReportDegradation;
}

export interface KPIResponse {
  success: boolean;
  kpis: { title: string; value: number | string; icon?: string; color?: string }[];
}

/** 环形图数据项 */
export interface RingChartItem {
  name: string;
  value: number;
}

/** 环形图配置 */
export interface RingChartConfig {
  title: string;
  data: RingChartItem[];
}

/** AI 大屏布局响应 */
export interface EChartsAiLayoutResponse {
  success: boolean;
  recommended_template: string;
  reason: string;
  block_title: string;
  nav_tabs?: string[];
  ring_charts?: RingChartConfig[];
  charts: EChartItem[];
}

/* ===== V2 分析引擎类型 ===== */

/** AI 返回的分析意图 */
export interface AnalysisIntent {
  business_question: string;
  analysis_goal: string;
  priority: 'high' | 'medium' | 'low';
  reason: string;
}

/** KPI 指标项（V2） */
export interface PackageKPIItem {
  label: string;
  value: string;
  change: string | null;
  kpi_type: 'sum' | 'avg' | 'count' | 'rate' | 'change';
}

/** 画像总览表的单元格结构（后端 RenderedCell） */
export interface TableCellData {
  value?: unknown;
  rank?: number;
  direction?: string;   // good(绿)/equal(黄)/bad(红)/neutral(不染色)
  cell_type?: string;   // number/percentage/category/neutral/text
  highlight?: boolean;
  count?: number;
}

/** 表格数据（V2） */
export interface PackageTableData {
  title: string;
  table_type: 'summary' | 'ranking' | 'cross' | 'growth' | 'correlation' | 'detail' | 'exception' | 'profile_overview';
  columns: string[];
  rows: unknown[][];
  /** 画像总览表的区块/模块元数据（仅 profile_overview 表使用） */
  chart_config?: {
    blocks?: { title: string; keys: string[] }[];
    module?: string;
    feature_cols?: string[];
  };
}

/** 图表项（V2） */
export interface PackageChartItem {
  slot: string;
  chart_type: string;
  title: string;
  role: 'primary' | 'secondary' | 'detail';
  option: Record<string, unknown>;
  /** 原始扁平 rows（ChartData.data），供前端模板库组件使用（同期群/气泡/表格等吃扁平清单的组件） */
  raw_data?: Record<string, unknown>[];
}

/** 分析包（全系统统一数据对象） */
export interface AnalysisPackage {
  id: string;
  analysis_type: string;
  business_question: string;
  algorithm: string;
  dimension: string;
  metric: string;
  kpis: PackageKPIItem[];
  tables: PackageTableData[];
  charts: PackageChartItem[];
  insights: string[];
  conclusions: string[];
  can_run: boolean;
  fallback_from: string | null;
  suggestion?: string;
  saved_at: string | null;
  data_profile: Record<string, string[]>;
}

/** /insights/generate 响应（V2） */
export interface InsightsV2Response {
  success: boolean;
  insights: string;
  intents: AnalysisIntent[];
}

/** /analysis/run 响应 */
export interface AnalysisRunResponse {
  packages: AnalysisPackage[];
}

/** /analysis/save 响应 */
export interface AnalysisSaveResponse {
  saved_count: number;
  package_ids: string[];
}

/** /dashboard/saved-packages 响应 */
export interface SavedPackagesResponse {
  success: boolean;
  packages: AnalysisPackage[];
  total: number;
}

/** V5: Card 驱动 BI 大屏 */
export interface CardItem {
  id: string;
  type: 'kpi' | 'chart' | 'table' | 'insight' | 'warning' | 'fallback';
  title: string;
  priority: number;
  size: 's' | 'm' | 'l' | 'xl';
  score: number;
  data: Record<string, unknown>;
  chart_type?: string;
  fallback_chain?: Array<Record<string, unknown>>;
}

export interface CardMeta {
  total_cards: number;
  insight_strength: number;
  data_quality: number;
}

/* ===== 多数据集管理类型 ===== */

export interface DatasetInfo {
  dataset_id: string;
  file_name: string;
  file_size_bytes: number;
  rows: number;
  columns: string[];
  column_info: {
    name: string;
    dtype: string;
    missing: number;
    missing_rate: number;
    unique: number;
    sample: string;
  }[];
  preview: Record<string, unknown>[];
  uploaded_at: number;
  is_active?: boolean;
  // 多表合并宽表标记（合并生成的宽表才有）
  is_merged?: boolean;
  sources?: string[];   // 来源 dataset_id 列表
  merge_keys?: string[]; // 实际使用的关联键列名
}

export interface DatasetListResponse {
  success: boolean;
  datasets: DatasetInfo[];
  used_bytes?: number;
  quota_bytes?: number;
  /** 该用户已上传的"数据集数" - 与字节配额并行 */
  dataset_count?: number;
  /** 单用户最大数据集数（登录用户才有，None 表示不限制） */
  dataset_limit?: number | null;
  // AI 配置（整 session 一份，随 /data/datasets 一并拉回，放在 response 根级）
  api_key?: string;
  ai_provider?: string;
  custom_model?: string;
  custom_base_url?: string;
}

/** /analysis/process-datasets 提交响应 */
export interface ProcessSubmitResponse {
  task_id: string;
  total: number;
}

/** 单个数据集的处理状态 */
export interface DatasetProcessStatus {
  status: 'pending' | 'running' | 'done' | 'error';
  pkg_count?: number;
  error?: string;
  // 该数据集产出的分析包（含 charts/option），供前端直接渲染
  packages?: AnalysisPackage[];
  // 合并宽表处理项携带的元信息
  kind?: 'single' | 'merged';
  sources?: string[];
  merge_keys?: string[];
}

/** /analysis/process-datasets/status/{task_id} 轮询响应 */
export interface ProcessStatusResponse {
  status: 'running' | 'done' | 'error';
  total: number;
  completed: number;
  datasets: Record<string, DatasetProcessStatus>;
  // 整个任务失败时的顶层错误（如所有数据集处理异常）
  error?: string;
}

/** /clean/ai-clean 提交响应（异步：立即返回任务号） */
export interface AICleanSubmitResponse {
  task_id: string;
  total: number;
}

/** 单个数据集的 AI 清洗状态 */
export interface DatasetAICleanStatus {
  status: 'pending' | 'running' | 'done' | 'error';
  kind?: 'single' | 'merged';
  sources?: string[];
  merge_keys?: string[];
  explanation?: string;
  steps_applied?: Array<{ step: string; reason: string; success: boolean }>;
  rows_change?: number;
  note?: string;
  error?: string;
}

/** /clean/ai-clean/status/{task_id} 轮询响应 */
export interface AICleanStatusResponse {
  status: 'running' | 'done' | 'error' | 'partial';
  total: number;
  completed: number;
  datasets: Record<string, DatasetAICleanStatus>;
  error?: string;
}

/* ===== 认证 / 用户 / 历史类型 ===== */

export interface AuthUser {
  id: number;
  username: string;
}

export interface AuthResult {
  token: string;
  user: AuthUser;
}

export interface HistoryDataset {
  dataset_id: string;
  file_name?: string;
  original_path?: string;
  created_at?: number;
  package_count?: number;
  [key: string]: unknown;
}

export interface HistoryPackage {
  package_id: string;
  title?: string;
  description?: string;
  saved_at?: string;
  created_at?: number;
  [key: string]: unknown;
}
