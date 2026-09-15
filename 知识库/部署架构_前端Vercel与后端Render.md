---
title: 部署架构_前端Vercel与后端Render
aliases: [部署架构, 上线部署, Vercel, Render, 冷启动]
tags: [meta, tech/deploy, 后端, 前端, 总览]
created: 2026-09-15
---

# 部署架构：前端 Vercel + 后端 Render

> 回答一个问题：**能不能把整个项目（前端+后端）都署到 Render，让别人用 Render 给的一个网址打开？** 结论是「技术上能，而且后端代码早就写好了；但你要的『单网址给别人访问』现在架构已经满足」。相关总览见 [[项目架构全景图]]、[[前端技术栈]]、[[后端技术栈]]，索引见 [[README]]。

## 一、当前实际部署形态

两个平台分工，但**使用者只接触一个网址（Vercel 的）**：

| 角色 | 平台 | 网址 / 标识 |
|---|---|---|
| 前端（静态站） | Vercel | 别人访问的就是这个地址 |
| 后端（FastAPI） | Render | `https://data-analysis-0bfa.onrender.com` |

连接它们的那根线在 `frontend/.env.production`：

```
VITE_API_BASE=https://data-analysis-0bfa.onrender.com/api
VITE_MAX_UPLOAD_SIZE_MB=30
```

前端页面里的 API 请求发到这里。**别人从头到尾不需要知道 Render 的存在**——所以「让朋友用一个网址打开」这件事，现在就成立，不需要为此改架构。

## 二、Render 侧的配置（`render.yaml`，已入库）

```yaml
type: web
name: data-analysis
runtime: python
plan: free
buildCommand: pip install -r backend/requirements.txt
startCommand: cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
healthCheckPath: /api/health
autoDeploy: true
```

注意点：**Root Directory 必须留空**（整个仓库根），因为它自己 `cd backend`。`autoDeploy: true` 意味着**推到 GitHub 就自动触发部署**——这也是为什么「未验证的代码不能随便 push」。

## 三、为什么本地开发不会误连线上

很多人担心跑本地时打到生产。实际链路是安全的：

1. Vite **dev 模式**加载 `.env.local` + `.env.development`，**不加载** `.env.production`（那是 build 才用）；
2. `.env.local` 里没设 `VITE_API_BASE` → `import.meta.env.VITE_API_BASE` 为 undefined；
3. 代码回退成相对路径：`frontend/src/api/client.ts` 第 16 行 `let API_BASE = import.meta.env.VITE_API_BASE || '/api'`；
4. Vite proxy 把 `/api` 转到本机 8001（`frontend/vite.config.ts` 的 `server.proxy`）。

**结论：本地 `npm run dev` + 本地后端 = 打到本地，测的是本地代码。** 详见 [[Vite是什么]]、[[前端怎么调后端API]]。

## 四、后端其实已经内置了「全栈托管」能力

`backend/main.py` L147–192 早就写好了一整套静态托管，不用新写代码：

| 行号 | 能力 |
|---|---|
| 149–163 | `_resolve_frontend_dist()`：在 4 个候选路径找 `frontend/dist`，兼容不同部署目录结构 |
| 166–173 | `GET /`：dist 存在就返回 `index.html`（且禁止 CDN 缓存），否则返回健康检查 JSON |
| 178–192 | `GET /{full_path:path}` SPA 回退：先找静态文件，找不到回退 `index.html`（**刷新子页面不会 404**）；并跳过 `api/` 前缀，避免吞掉真实 API |
| 189–190 | 带 hash 的 JS/CSS/图片缓存 1 年 (`immutable`) |
| 81–87 | CORS `allow_origins=["*"]` |

**只要 Render 上存在 `frontend/dist`，后端自己就会托管前端，一个服务搞定。**

## 五、真要全栈迁到 Render，三个卡点

### 卡点 1：`dist` 从哪来（最麻烦，存在规则打架）

`main.py` 的注释写的是「随仓库提交 `frontend/dist`」，但 `.gitignore` 第 76 行明确忽略 `frontend/dist/`。两条规则矛盾。两条解法：

| 方案 | 做法 | 风险 |
|---|---|---|
| A. Render 自己构建 | 改 `render.yaml` 的 `buildCommand` 跑 `npm ci && npm run build` | 依赖 Render 的 Python 运行环境里有 Node —— **未实测，不能打包票** |
| B. 本地构建后提交 dist | 取消 gitignore，把 dist 强推上去 | 百分百能跑，但构建产物混进仓库；本项目带 163 张 jpg，仓库会变肿 |

### 卡点 2：冷启动（**免费版的命门，改架构解决不了**）

Render 免费 Web Service：**15 分钟无流量即休眠**，下一个访客要等 **30–60 秒**才起来，页面长时间白屏，容易被误认为坏了。

关键在于：**这一点两种架构都一样**。即使前端留在 Vercel，它背后调的还是 Render，第一次调 API 照样要等 Render 醒。所以这不是「搬到 Render」的问题，是**免费额度**的问题。

唯一根治办法是付费 Starter（去掉休眠 + 更多内存），价格以 Render 官网为准。

### 卡点 3：一个要改的小地方

真做同源全栈，`frontend/.env.production` 里写死的 Render 地址应改成 `/api` 让请求走同源，否则仍然跨域绕一圈（CORS 允许 `*` 所以能跑通，但多此一举且绑死那个域名）。

### 附带：内存限制

Render 免费版 **512MB 内存**，多人同时上传 CSV 跑 pandas 分析容易卡死/502。要有心理预期。

## 六、决策记录（2026-09-15）

| 事项 | 用户决定 |
|---|---|
| 部署路线 | **先本地冒烟通过再部署**，不把未验证代码推上线（涉及 SQLite/删 4 页面/改 6 组件/新增 `llm_providers.py`） |
| 冷启动 | **先接受免费版**，交付时提醒对方「第一次打开要等几十秒」 |
| Render 仓库 | 维持现仓库 `git@github.com:linziquan/double_data_analysis.git`（SSH，22 端口通；HTTPS 443 被挡） |

> 铁律：**未经本地端到端验证（尤其 AI 报告生成链路），不执行 push**。因为 `autoDeploy: true`，push 即上线。
