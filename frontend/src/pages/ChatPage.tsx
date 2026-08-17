/* DataMind AI - 聊天分析页
 * 模型来源：用户在「API 配置」页选择的服务商 + 自定义模型。
 * 后端 /api/chat/send 接收前端透传的 api_key/ai_provider/custom_model/custom_base_url，
 * 任一字段为空时退回 DataAnalysisAgent 默认值（Agnes）。
 * 数据上传走左侧「数据上传」页面；本页不再内嵌上传入口。
 * 预留 message.kind='choice' 渲染位（大脑方后续把工具 options 以选择框形式推回）。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Send, Database, AlertTriangle, MessageSquare, Sparkles, Loader2, Info } from 'lucide-react';
import { useData } from '../contexts/DataContext';
import { listDatasets, chatSend, getChatMessages, selectDataset } from '../api/client';
import { marked } from 'marked';
import EtherealChart from '../components/EtherealCharts/EtherealChart';
import ReportCard from '../components/ReportCard';
import BigScreenCard from '../components/BigScreenCard';
import { AI_PROVIDERS } from '../contexts/DataContext';

// 与 AnalysisPage / VisualizationRenderer 一致：后端 AI 输出为 Markdown（可信源），渲染成富文本
function renderMarkdown(text: string): string {
  return marked.parse(text || '') as string;
}



// 把 LLM 写的简单 chart 结构（{chart_type,x,y,data,title}）转成仙气组件认识的 chartNode
function adaptChartToNode(chart: any) {
  // 完整 ECharts option（generate_chart 产出，含 series 字段）→ 直接透传，不转换
  if (chart?.series && Array.isArray(chart.series)) {
    return chart;
  }
  // 以下为 LLM 简单格式兼容（execute_python 产出）
  const t = chart?.chart_type;
  if (t === 'bar' || t === 'line') {
    return { xAxis: { data: chart.x || [] }, series: [{ type: t, data: chart.y || [] }], title: chart.title };
  }
  if (t === 'pie') {
    return { series: [{ type: 'pie', data: (chart.data || []).map((d: any) => ({ name: d.维度, value: d.数值 })) }], title: chart.title };
  }
  if (t === 'ranking') {
    return { data: chart.data || [], title: chart.title };
  }
  if (t === 'table') {
    // EtherealTable 只读 node.rows + node.columns，不认 node.data
    return { rows: chart.data || [], columns: Object.keys((chart.data && chart.data[0]) || {}), title: chart.title };
  }
  return chart; // 其余类型透传兜底
}

// 判定一个 series.data / data 是否"实际有内容"。
// 避免把全 0 全空数组误判为可渲染（用户截图里的 "Y轴1/0.8/.../0、X轴0/0" 的怪图就是这样产生的）。
function hasMeaningfulData(arr: any[]): boolean {
  if (!Array.isArray(arr) || arr.length === 0) return false;
  return arr.some((v: any) => {
    // 对象格式（{value:0}）取 value；数组/标量直接判定
    const n = typeof v === 'object' && v !== null ? (v as any).value : v;
    if (n === null || n === undefined || n === '') return false;
    const num = typeof n === 'number' ? n : Number(n);
    return Number.isFinite(num) && num !== 0;
  });
}

// 判定一张图表是否真正"出图了"。
// 既看 series.data 长度，也看里面有没有非零值；空/全 0 视为空图。
function chartIsRenderable(chart: any): boolean {
  if (!chart) return false;
  const seriesData = chart.series?.[0]?.data;
  if (Array.isArray(seriesData)) {
    if (chart.series.length === 1) {
      return hasMeaningfulData(seriesData);
    }
    // 多 series：任一有数据即可（堆叠/多指标也算）
    return chart.series.some((s: any) => hasMeaningfulData(s?.data));
  }
  if (Array.isArray(chart.data)) {
    return hasMeaningfulData(chart.data);
  }
  return false;
}

// 从一个 tool result 中找"第一个真正能渲染的图表"
// （generate_chart 顶层 data.chart，以及 run_template/run_analysis 包内 charts[*].option）。
function pickRenderableChart(tr: ToolResult): { node: any; chartType: string } | null {
  // 1) 顶层 data.chart
  let raw: any = tr.data?.chart;
  if (raw && !Array.isArray(raw.series)) {
    raw = adaptChartToNode(raw);
  }
  const t1 = raw?.chart_type || raw?.series?.[0]?.type || (tr.data?.packages?.[0] as any)?.type;
  if (chartIsRenderable(raw)) {
    return { node: raw, chartType: t1 || 'unknown' };
  }

  // 2) packages[*].charts[*].option / charts[*]
  const pkgs: any[] = tr.data?.packages || tr.data?.full_packages || [];
  for (const pkg of pkgs) {
    const charts: any[] = Array.isArray(pkg?.charts) ? pkg.charts : [];
    for (const c of charts) {
      const node = c?.option || c;
      const normalized = node && !Array.isArray(node.series) ? adaptChartToNode(node) : node;
      if (chartIsRenderable(normalized)) {
        return { node: normalized, chartType: c?.chart_type || normalized?.series?.[0]?.type || 'unknown' };
      }
    }
  }
  return null;
}

// 单个工具执行结果行：图表/报告/大屏按工具类型内联渲染，不藏按钮后
function ToolResultRow({ tr }: { tr: ToolResult }) {
  const isOk = tr.status === 'ok';

  // 报告 / 大屏：整块内联渲染（数据在 data.report / data.bigscreen，无 chart 字段）
  if (tr.tool === 'generate_report' && tr.data?.report) {
    return <ReportCard report={tr.data.report} />;
  }
  if (tr.tool === 'build_dashboard' && tr.data?.bigscreen) {
    return <BigScreenCard bigscreen={tr.data.bigscreen} />;
  }

  const picked = pickRenderableChart(tr);
  const badge = isOk ? 'bg-emerald-500/15 border border-emerald-400/40 text-emerald-700'
                     : 'bg-rose-500/15 border border-rose-400/40 text-rose-700';
  // 有图表时只渲染可视化本体，不显示「工具名 + 执行成功」这种后台徽章；
  // 折叠摘要（"已分析 · 点开看执行过程"）已经说明该步骤完成，没必要在每条工具下再重复"执行成功"。
  // 报告/大屏已在前面单独渲染，不会走到这里。
  if (picked) {
    return (
      <EtherealChart
        chartType={picked.chartType}
        chartNode={picked.node}
        height={360}
      />
    );
  }
  return (
    <div className={`text-xs rounded-lg px-2.5 py-1.5 ${badge}`}>
      <div className="flex items-start gap-2">
        <span className="font-medium shrink-0">{tr.tool}</span>
        <span className="opacity-70">{isOk ? '执行成功' : '执行失败'}</span>
      </div>
    </div>
  );
}

type Role = 'user' | 'assistant';
interface ChoiceOption {
  id: string;
  label: string;
  description?: string;
}
interface ToolResult {
  tool: string;
  status: string;
  summary?: string;
  data?: any;
}
interface DataPreview {
  rows?: number;
  columns?: string[];
  head?: Array<Record<string, any>>;
}
interface ChatMsg {
  role: Role;
  content: string;
  kind?: 'text' | 'choice';
  choices?: ChoiceOption[];        // kind='choice' 时的选择按钮
  toolResults?: ToolResult[];      // 本轮工具执行结果
  dataPreview?: DataPreview | null; // 清洗后数据预览
  /** 该条助手消息携带待选择的清洗方案时，用户是否已点选（避免重复点） */
  choiceResolved?: boolean;
  /** 请求进行中、尚未返回内容的占位态（渲染"AI 正在分析数据…"加载条） */
  pending?: boolean;
}

export default function ChatPage() {
  const { state, dispatch, ensureValidSession } = useData();
    const { sessionId, datasets, activeDatasetId } = state;
    // 当前对话使用的模型名：用户在「API 配置」页自填的优先；空时回落到服务商默认；
    // 兜底是 Agnes（与后端 DataAnalysisAgent 默认值一致）。
    const displayModel = useMemo(() => {
      const provider = (state.aiProvider || 'agnes').toLowerCase();
      const preset = AI_PROVIDERS.find((p) => p.id === provider);
      const presetDefault = preset?.model ?? 'agnes-2.0-flash';
      return (state.customModel || '').trim() || presetDefault;
    }, [state.aiProvider, state.customModel]);
    const [messages, setMessages] = useState<ChatMsg[]>(() => {
      // 优先读 sessionStorage 缓存：避免切走切回 / 路由跳转 unmount→remount 后
      // 整页对话被清空（用户认为"切回就丢"）。
      // 按 sessionId 做 key，不同会话不会串。
      const cached = typeof window !== 'undefined' && sessionId
        ? sessionStorage.getItem(`chat:msgs:${sessionId}`)
        : null;
      if (cached) {
        try { return JSON.parse(cached); } catch { return []; }
      }
      return [];
    });
    const [input, setInput] = useState('');
    const [sending, setSending] = useState(false);
    // 用 ref 记录上次同步过的 sid，避免组件内 useState 排在 useEffect 之后
    // （违反 React hooks 顺序规则，会导致 lastSyncedSid 状态错位，
    // 进而 sessionId 变化时早 return 跳过拉取 → 用户切回 ChatPage 看不见对话）。
    const lastSyncedSidRef = useRef<string | null>(null);
    const bottomRef = useRef<HTMLDivElement>(null);

    const hasData = datasets.length > 0;

    useEffect(() => {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // 进入页面时同步一次数据集列表（与 UploadPage 一致）
    useEffect(() => {
      let alive = true;
      (async () => {
        try {
          const res = await listDatasets(sessionId);
          if (!alive) return;
          dispatch({ type: 'SET_DATASETS', datasets: res.datasets });
        } catch { /* 无数据忽略 */ }
      })();
      return () => { alive = false; };
    }, [sessionId, dispatch]);

    // 进入对话页时回填历史会话：从历史会话点进来后，把 session.messages
    // （user/assistant 纯文字对话流）灌入，恢复上次对话记录。
    // 同时：sessionId 变化时先清空对话窗口，避免之前会话的消息
    // 串到下一个 session 里的尴尬。
    // 普通的「登录后新建 session」用 `assign_new_session_to_user` 会得到一个
    // 全新 session（messages 为空），所以这里清空是安全的。
    // 注意：这里用 ref 而不是 useState 来追踪 lastSyncedSid，避免 hooks 顺序错位
    // （原先的 useState 排在了 useEffect 之后，违反规则）。
    useEffect(() => {
        if (!sessionId) return;
        if (lastSyncedSidRef.current === sessionId) return;
        lastSyncedSidRef.current = sessionId;
        let alive = true;
        // 尝试读 sessionStorage 缓存以秒级恢复（仅当 React state 已为空时，
        // 避免覆盖用户已经在本页面看到/编辑中的消息）。
        setMessages((prev) => {
          if (prev.length > 0) return prev; // 已有内容（用户新对话中），不动
          const cached = sessionStorage.getItem(`chat:msgs:${sessionId}`);
          if (cached) {
            try { return JSON.parse(cached); } catch { return []; }
          }
          return prev;
        });
        (async () => {
          // 最多重试 8 次：刚恢复的会话里 session_manager 仍在 hydrate（从 DB 重建），
          // 第一次调 /chat/messages 拿到空时不要直接放弃，否则用户进入历史会话
          // 却看不到自己之前问过的对话。每次间隔 700ms，整体 ~5.6s 窗口。
          for (let attempt = 0; attempt < 8; attempt++) {
            if (!alive) return;
            try {
              const res = await getChatMessages(sessionId);
              const hist: ChatMsg[] = (res.messages || [])
                .filter((m: any) => m.role === 'user' || m.role === 'assistant')
                .map((m: any) => ({ role: m.role, content: m.content || '' }));
              if (hist.length) {
                setMessages(hist);
                return;
              }
              // 拿到空结果：等 hydrate 完成再重试。
              if (attempt < 7) await new Promise((r) => setTimeout(r, 700));
            } catch { /* 无历史忽略 */ }
          }
        })();
        return () => { alive = false; };
      }, [sessionId]);

      // 把当前会话的消息写入 sessionStorage 缓存。
        // 这样切走切回 ChatPage（路由 unmount→remount）或临时刷新，
        // 都能秒级恢复对话，不依赖网络 /chat/messages 拉取。
        // 注意：只在 sessionId 已知时缓存；切走不同会话也不会被覆盖到旧 sid。
        // 同时过滤掉 pending=true 的占位消息，避免刷新后"AI 正在分析…"加载条永远卡住。
        useEffect(() => {
          if (!sessionId) return;
          try {
            const safe = messages.filter((m) => !m.pending);
            sessionStorage.setItem(`chat:msgs:${sessionId}`, JSON.stringify(safe));
          } catch {
            // sessionStorage 可能因隐私模式/配额限制抛错，忽略即可
          }
        }, [sessionId, messages]);

  // 顶栏下拉切换数据集：调用后端 selectDataset，并同步本地 activeDatasetId
  const handleSelectDataset = useCallback(async (datasetId: string) => {
    if (!sessionId) return;
    try {
      await selectDataset(sessionId, datasetId);
    } catch (e) {
      // 后端失败不影响本地视图；下次请求会用最新 active
      console.warn('[ChatPage] selectDataset failed', e);
    }
    dispatch({ type: 'SELECT_DATASET', datasetId });
  }, [sessionId, dispatch]);

  const send = useCallback(async (choiceId?: string) => {
  // 防御：choiceId 必须是字符串，否则（如对象/undefined 经异常路径传入）
  // 会被 String() 成 "[object Object]" 发给后端导致误执行清洗。非字符串一律视为无选择。
  if (typeof choiceId !== 'string') choiceId = '';
  // choiceId 非空：复用上一条助手消息的原文作为提问，回传用户选择
  const safeChoiceId = choiceId ? String(choiceId) : '';
  const text = safeChoiceId ? '' : input.trim();
  // 上一轮请求还在跑：静默忽略会让用户困惑（文字卡在框里），给出轻量提示后返回
  if (!safeChoiceId && sending) {
    setMessages((m) => [...m, {
      role: 'assistant',
      content: '⏳ 上一轮分析还在进行中，请稍候再发消息。',
      pending: false,
    }]);
    return;
  }
  if (!safeChoiceId && !text) return;
  if (!hasData) {
    setMessages((m) => [...m, {
      role: 'assistant',
      content: '⚠️ 当前会话还没有数据，请先到「数据上传」页面上传文件后再来对话。',
      pending: false,
    }]);
    return;
  }

  if (!safeChoiceId) setInput('');
  setSending(true);

    // 用户点选方案 → 先把该选择作为一条 user 消息展示
    if (safeChoiceId) {
      setMessages((m) => [...m, { role: 'user', content: `▶ 选择：${safeChoiceId}` }]);
    } else {
      setMessages((m) => [...m, { role: 'user', content: text }]);
    }

    // 先推一条 pending 占位助手消息，渲染"AI 正在分析数据…"加载条
    setMessages((m) => [...m, { role: 'assistant', content: '', pending: true }]);

    const replaceLast = (msg: ChatMsg) =>
      setMessages((m) => m.map((x, idx) => (idx === m.length - 1 ? msg : x)));

    try {
      const r = await chatSend(
        sessionId,
        safeChoiceId ? '' : text,
        safeChoiceId,
        {
          apiKey: state.apiKey,
          aiProvider: state.aiProvider,
          customModel: state.customModel,
          customBaseUrl: state.customBaseUrl,
        },
      );
      replaceLast({
        role: 'assistant',
        content: r.content ?? '（无回复）',
        kind: r.kind,
        choices: r.choices ?? [],
        toolResults: r.tool_results ?? [],
        dataPreview: r.data_preview ?? null,
      });
    } catch (e: any) {
      const err = e?.response?.data?.detail || e?.message || 'AI 调用失败';
      replaceLast({ role: 'assistant', content: `⚠️ ${err}` });
    } finally {
      setSending(false);
    }
  }, [input, sending, hasData, sessionId]);

  // 用户点击清洗方案按钮
  const onChoose = useCallback((choiceId: string, msgIndex: number) => {
    // 防御：choiceId 非字符串则直接忽略，绝不把对象传入 send（避免 [object Object]）
    if (typeof choiceId !== 'string') return;
    // 标记该条助手消息已解决，避免重复点
    setMessages((m) => m.map((msg, i) =>
      i === msgIndex ? { ...msg, choiceResolved: true } : msg));
    send(choiceId);
  }, [send]);

  return (
    <div className="relative h-screen">
      <div className="bg-layer" />
      <div className="relative z-10 h-full flex flex-col bg-transparent text-slate-800">
      {/* 顶栏 */}
      <header className="flex items-center gap-3 px-6 py-4 border-b border-white/40">
        <Sparkles className="w-5 h-5 text-violet-500" />
        <h1 className="text-lg font-semibold text-slate-800">DataMind AI 对话分析</h1>
        <span className="ml-2 text-xs px-2 py-0.5 rounded-full bg-violet-500/15 text-violet-700 border border-violet-400/40">
          模型：{displayModel}
        </span>
        <div className="ml-auto flex items-center gap-2 text-xs text-slate-500">
          <span
            className="group relative inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-100/70 border border-slate-200/60 text-slate-500 cursor-help"
            title="每个分析工具在同一轮对话内有少量调用上限（防失控熔断）。若 AI 给到前置概览就停了，可继续追问具体维度（如：按月看趋势 / 按类目看排名）引导它再调维度分析工具。"
          >
            <Info className="w-3 h-3" />
            <span>工具调用上限说明</span>
          </span>
          <Database className="w-4 h-4" />
          {hasData ? (
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-xs text-slate-400 shrink-0">
                共 {datasets.length} 张表
                {datasets.some((d) => d.is_merged) ? (
                  <>
                    {' · '}
                    <span className="text-violet-600">含 {datasets.filter((d) => d.is_merged).length} 张合并宽表</span>
                  </>
                ) : datasets.length >= 2 ? (
                  <>
                    {' · '}
                    <span className="text-amber-600">未检测到可关联键（未生成宽表）</span>
                  </>
                ) : null}
              </span>
              <div className="flex items-center gap-1.5 min-w-0 flex-wrap">
                {datasets.map((d) => {
                  const active = (activeDatasetId || datasets[0]?.dataset_id) === d.dataset_id;
                  return (
                    <button
                      key={d.dataset_id}
                      onClick={() => handleSelectDataset(d.dataset_id)}
                      className={
                        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs border transition ' +
                        (active
                          ? 'bg-violet-500/15 border-violet-400/60 text-violet-700 font-medium'
                          : 'bg-white/60 border-slate-200/70 text-slate-600 hover:border-violet-300 hover:bg-violet-500/5')
                      }
                      title={d.is_merged
                        ? `合并宽表（来自 ${(d.sources ?? []).join(', ')}；关联键 ${(d.merge_keys ?? []).join(', ')}）`
                        : '点击切换为当前分析的数据集'}
                    >
                      <Database className="w-3 h-3" />
                      {d.is_merged ? <span>🔗</span> : null}
                      <span className="truncate max-w-[180px]">{d.file_name}</span>
                      {d.rows ? <span className="text-slate-400">· {d.rows}</span> : null}
                    </button>
                  );
                })}
              </div>
            </div>
          ) : (
            <span className="text-slate-400">（当前会话暂无数据，请到「数据上传」页面上传）</span>
          )}
        </div>
      </header>

      {/* 消息区 */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center text-slate-500 gap-3">
            <MessageSquare className="w-12 h-12 text-violet-400/70" />
            {hasData ? (
              <>
                <p className="max-w-md text-slate-600">
                  已上传 <b>{datasets.length}</b> 个数据集{datasets.length > 1 ? '（可在左上角下拉切换）' : ''}，向我提问吧。例如「有哪些列缺失？」「帮我做个趋势分析」{datasets.length > 1 ? '，或说「把这几张表合并关联分析」' : ''}。
                </p>
                <p className="text-xs text-slate-400">（这个历史会话还没生成过对话记录；开始提问后会自动保留在这里。）</p>
              </>
            ) : (
              <div className="max-w-md text-slate-600">
                <p>当前会话还没有数据，请先到「数据上传」页面上传文件后再回来对话。</p>
                <p className="mt-2 text-xs text-slate-400">（数据上传、清洗都在左侧「数据上传」页面里完成。）</p>
              </div>
            )}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[82%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
              m.role === 'user'
                ? 'bg-violet-500/15 border border-violet-400/40 text-slate-800 whitespace-pre-wrap'
                : 'bg-white/60 border border-white/70 text-slate-800 backdrop-blur-sm'
            }`}>
              {m.pending ? (
                <div className="flex items-center gap-2 text-slate-500">
                  <Loader2 className="w-4 h-4 animate-spin text-violet-500" />
                  <span>AI 正在分析数据…</span>
                </div>
              ) : m.role === 'user' ? (
                m.content
              ) : (
                <div className="md-body" dangerouslySetInnerHTML={{ __html: renderMarkdown(m.content) }} />
              )}

              {/* 工具执行结果：可视化结果（图表/报告/大屏）平铺为正文，纯文字结果收进折叠区 */}
              {m.toolResults && m.toolResults.length > 0 && (() => {
                const ok = m.toolResults.filter((tr: ToolResult) => tr.status !== 'fail');
                const visualResults = ok.filter(
                  (tr: ToolResult) =>
                    // 图/报告/大屏平铺：含 data.chart、data.report、data.bigscreen，
                    // 以及 run_template/run_analysis 包内 charts[*].option（funnel 等），剔除全 0/空系列空图
                    tr.data?.chart ||
                    tr.data?.report ||
                    tr.data?.bigscreen ||
                    pickRenderableChart(tr) !== null,
                );
                const otherResults = ok.filter(
                  (tr: ToolResult) => pickRenderableChart(tr) === null &&
                    !(tr.data?.chart || tr.data?.report || tr.data?.bigscreen),
                );
                return (
                  <>
                    {visualResults.length > 0 && (
                      <div className="mt-3">
                        {visualResults.map((tr, ti) => (
                          <ToolResultRow key={ti} tr={tr} />
                        ))}
                      </div>
                    )}
                    {otherResults.length > 0 && (
                      <details className="mt-3 group">
                        <summary className="cursor-pointer select-none text-xs text-slate-500 hover:text-violet-600 flex items-center gap-1.5">
                          <span className="text-emerald-500">✓</span>
                          已分析 · 点开看执行过程
                        </summary>
                        <div className="mt-2 space-y-1.5">
                          {otherResults.map((tr, ti) => (
                            <ToolResultRow key={ti} tr={tr} />
                          ))}
                        </div>
                      </details>
                    )}
                  </>
                );
              })()}

              {/* 清洗后数据预览 */}
              {m.dataPreview && m.dataPreview.head && m.dataPreview.head.length > 0 && (
                <div className="mt-3 overflow-x-auto rounded-lg border border-slate-200/60">
                  <table className="text-xs">
                    <thead>
                      <tr className="bg-white/70">
                        {(m.dataPreview.columns || Object.keys(m.dataPreview.head[0])).map((c) => (
                          <th key={c} className="px-2.5 py-1.5 text-left font-medium text-slate-700 whitespace-nowrap">{c}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {m.dataPreview.head.map((row, ri) => (
                        <tr key={ri} className="border-t border-slate-200/50">
                          {(m.dataPreview.columns || Object.keys(row)).map((c) => (
                            <td key={c} className="px-2.5 py-1.5 text-slate-600 whitespace-nowrap">{String(row[c] ?? '')}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* 清洗方案选择按钮 */}
              {m.choices && m.choices.length > 0 && !m.choiceResolved && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {m.choices.map((c) => (
                    <button
                      key={c.id}
                      onClick={() => onChoose(c.id, i)}
                      disabled={sending}
                      className="text-left px-3 py-2 rounded-xl bg-violet-500/15 border border-violet-400/50 text-violet-700 hover:bg-violet-500/25 disabled:opacity-50 transition-colors"
                    >
                      <div className="text-sm font-medium">{c.label}</div>
                      {c.description && <div className="text-xs text-violet-600/70 mt-0.5">{c.description}</div>}
                    </button>
                  ))}
                </div>
              )}
              {m.choices && m.choices.length > 0 && m.choiceResolved && (
                <div className="mt-3 text-xs text-emerald-600">✓ 已执行清洗，等待结果…</div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* 输入区 */}
      <div className="px-6 py-4 border-t border-white/40">
        <div className="flex items-end gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
            rows={2}
            placeholder={hasData ? '输入你的问题，回车发送…' : '请先上传数据后再提问'}
            className="flex-1 resize-none rounded-xl bg-white/60 border border-slate-300/60 px-4 py-3 text-sm text-slate-800 outline-none focus:border-violet-400/60"
          />
          <button
            onClick={send}
            disabled={sending || !hasData}
            className="px-5 py-3 rounded-xl bg-violet-500/80 border border-violet-500 text-white hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center gap-2"
          >
            <Send className="w-4 h-4" /> {sending ? '思考中…' : '发送'}
          </button>
        </div>
        {!hasData && (
          <p className="mt-2 text-xs text-amber-600 inline-flex items-center gap-1">
            <AlertTriangle className="w-3 h-3" /> 当前会话没有数据，请先到「数据上传」页面上传文件。
          </p>
        )}
      </div>
      </div>
    </div>
  );
}
