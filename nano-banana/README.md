# nano-banana — 新词站点规划与落地执行文档

English version follows Chinese.

[English](#english) | [中文](#中文)

---

## 中文

### 1) 网站定位
- __主题__：围绕新涌现的图像编辑/重绘模型“nano-banana”（参考截图与业内传播），做“最快信息聚合 + 可验证对比 + 新手入门”的轻量内容站。
- __目标人群__：AI 绘画从业者、设计师、AIGC 爱好者、Prompt 工程师、产品/媒体从业者。
- __价值主张__：
  - 第一时间聚合「模型动态、使用教程、Prompt 范式、效果对比、最佳实践」。
  - 提供可复现的对比基准（基于固定测试图与固定参数），帮助用户判断“是否值得切换/尝试”。
  - 提供面向新手的“5 分钟上手路线”。

### 2) 域名与品牌
- 已注册：`nano-banana.live`、`nano-banana.online`。
- 建议：
  - 主站使用 `nano-banana.live`（强调“活跃/最新”）。
  - 备用/镜像/实验站使用 `nano-banana.online`。
  - 统一品牌名：Nano Banana（记法：NB）。

### 3) 网站信息架构（IA）
- __首页 `/`__：模型简介 + 最新动态 + 对比卡片预览 + 新手入口。
- __动态 `/news/[slug]`__：跟踪模型版本迭代、生态动向、兼容性、UI 插件等。
- __指南 `/guides/[slug]`__：安装/接入、参数解释、工作流（本地/云端/Colab/UI 工具）。
- __对比 `/benchmarks`__：
  - `/benchmarks` 列表页：不同任务场景的基准集合（重绘、人像修复、场景替换、风格迁移等）。
  - `/benchmarks/[task]` 详情：固定输入、固定参数、可复现流程、下载链接、原图/结果图对比（lightbox）。
- __Prompt 库 `/prompts`__：
  - `/prompts` 列表：主题分类、可复制、含参数注释。
  - `/prompts/[slug]` 详情：示例图、适用场景、注意事项、可视化参数块。
- __画廊 `/gallery`__：社区精选与官方示例，支持标签与来源标注（版权声明/署名）。
- __工具 `/tools`__：启动脚本、Colab、ComfyUI/Forge 节点、参数计算器等链接聚合。
- __常见问题 `/faq`__：兼容性、显存需求、出图失败排查、伦理与版权提醒。
- __关于 `/about`__：站点说明、免责声明、联系/投稿、RSS。

### 4) 内容模型（建议用 MDX 管理）
- `News`：`title`、`excerpt`、`date`、`tags`、`cover`、`sourceUrl`。
- `Guide`：`title`、`readingTime`、`steps[]`、`requirements`（GPU/VRAM/平台）、`downloads[]`。
- `Benchmark`：`task`、`inputs[]`、`params`、`pipeline`、`results[]`（含对照：原图、NB、其它模型） 、`replicateSteps`。
- `Prompt`：`title`、`prompt`、`negativePrompt`、`params`、`samples[]`、`notes`、`license`。
- `GalleryItem`：`title`、`author`、`source`、`tags`、`image`、`promptRef`、`license`。

### 5) SEO 与分发策略
- __URL 规范__：小写、短路径，中文内容 slug 使用拼音或英文；如 `/guides/quickstart`。
- __元信息模板__：`<title>`、`description`、OG/Twitter、canonical、JSON-LD（Article/Gallery）。
- __多语言__：先中英双语，`/en/...` 作为英文路径；`hreflang` 互链。
- __站点地图/Robots__：`/sitemap.xml` 每日更新；`/robots.txt` 允许抓取，基准中临时页可 `noindex`。
- __结构化数据__：文章用 `Article`，对比页可用 `Dataset`/`CreativeWork` 扩展。
- __RSS/Atom__：`/rss.xml` 输出 News 与 Guides。

#### SEO 目标关键词（English-first）
- Primary：`nano-banana`、`nano banana`、`nanobanana`、`nano-banana image editing model`
- Secondary：`text-guided image editing`、`image-to-image editing`、`local edits`、`inpainting`、`outpainting`、`background replacement`、`face restoration`、`character consistency`、`style transfer`、`reproducible benchmarks`
- Context/Compare：`LMArena`、`FLUX Kontext`、`FLUX Kontext vs nano-banana`
- Workflow：`ComfyUI workflow`、`Colab pipeline`、`prompt engineering for image editing`

### 6) 技术栈与部署
- __静态优先（Vercel）__：Next.js 15 App Router + MDX，采用 SSG/ISR；无需自建数据库。
- __图片与对比__：`next/image` + 轻量 lightbox，对比组件支持 A/B 滑块。
- __表单/订阅__：Buttondown/ConvertKit（无后端）或 Cloudflare Turnstile + 简易 API Route。
- __分析__：Vercel Analytics 或 Umami（自部署可选）。
- __性能__：预渲染、图片优化、懒加载、`dynamicParams=false` 的批量静态化。

### 7) 初始交付清单（MVP ≤ 2 天）
- __页面__：`/`、`/benchmarks`、`/benchmarks/portrait-editing`（示例基准）、`/guides/quickstart`、`/prompts`、`/about`、`/faq`。
- __组件__：对比滑块、参数卡片、复制按钮、Tag 过滤、面包屑。
- __内容__：
  - 1 篇 News：宣布/解析“nano-banana”模型与竞品关系。
  - 1 篇 Guide：5 分钟快速上手（含 Colab/本地两条路线）。
  - 1 个 Benchmark：固定人像编辑案例三方对比（NB vs FLUX Kontext 等）。
  - 5 条 Prompt：常见场景可直接复制。
- __SEO__：站点级模板、sitemap、robots、OG 资源。

### 8) 路由/URL 设计
- `/` 首页
- `/news/[slug]`
- `/guides/[slug]`
- `/benchmarks`
- `/benchmarks/[task]`
- `/prompts`
- `/prompts/[slug]`
- `/gallery`
- `/tools`
- `/faq`
- `/about`
- `/en/...` 英文镜像路径

### 9) 内容与法务
- __免责声明__：
  - 站点仅为信息与教育用途；示例图像版权归原作者/来源所有；请遵循对应许可协议。
  - 对比结果受参数、硬件、版本影响；请以复现实验为准。
- __署名与来源__：所有示例图/基准需标注来源与许可。

### 10) 运营与增长
- __捕捉趋势__：监控官方/社区渠道，设立“快讯模板”，30 分钟内完成发布。
- __邮件订阅__：每周简报；发布时触发 RSS + 邮件自动化。
- __社媒卡片__：统一 OG 图模板，自动化生成（必要时使用 `og.svg` + 动态标题）。

### 11) 开发里程碑
- __M1：信息骨架__（本仓库）
  - 初始化 Next.js + MDX 项目结构与基础 SEO。
  - 完成首页/基准页/指南页模板与示例内容。
- __M2：对比与画廊__
  - A/B 滑块、灯箱、标签筛选。
- __M3：多语言与订阅__
  - `/en` 版本、RSS、订阅对接。

### 12) 快速开始（待项目初始化后更新）
- `npm i`、`npm run dev`、`npm run build`、`npm run start`
- 环境变量：`NEXT_PUBLIC_SITE_URL=https://nano-banana.live`

 ### 13) 静态 MVP（已创建）
 为了“最快上线”，已在 `web/` 目录生成纯静态版本（英文默认，Google/科技风格），包含：
 - 页面：`/`、`/benchmarks/`、`/benchmarks/portrait-editing.html`、`/guides/quickstart.html`、`/prompts/`、`/faq.html`、`/about.html`、`/404.html`
 - 资源与 SEO：`/styles.css`、`/og.svg`、`/robots.txt`、`/sitemap.xml`、全页 canonical/OG/Twitter meta

 部署到 Vercel（最简）：
 1. 新建 Git 仓库并推送本项目。
 2. Vercel -> New Project -> Import -> 选择该仓库。
 3. 在 “Root Directory” 选择 `web/` 作为项目根，Framework 选 “Other”。
 4. Build Command 留空，Output Directory 留空（静态直出）。
 5. 部署完成后，在 Vercel 的 “Domains” 绑定 `nano-banana.live` 与 `nano-banana.online`，将你的域名 DNS 的 A/ALIAS/CNAME 指向 Vercel 提示的记录。

 上线后可立刻被抓取；随后可无缝迁移到 Next.js + MDX（保持相同 URL 结构）。

---

## English

### 1) What this site is
A lightweight, static-first hub around the emerging image editing model “nano-banana”: fastest updates, reproducible benchmarks, practical guides, and prompt recipes.

### 2) Domains & brand
- Domains: `nano-banana.live` (primary), `nano-banana.online` (mirror/experiments).

### 3) Information Architecture
- Home `/`: intro, latest news, preview of comparisons, quickstart.
- News `/news/[slug]`
- Guides `/guides/[slug]`
- Benchmarks `/benchmarks`, `/benchmarks/[task]`
- Prompts `/prompts`, `/prompts/[slug]`
- Gallery `/gallery`
- Tools `/tools`
- FAQ `/faq`
- About `/about`
- English mirror under `/en`.

### 4) Tech stack
- Next.js App Router + MDX, SSG/ISR on Vercel.
- next/image, A/B slider, lightbox.
- Optional newsletter via Buttondown; analytics via Vercel Analytics.

### 5) MVP checklist
- Pages: Home, Benchmarks, one benchmark detail, Quickstart, Prompts, About, FAQ.
- Content: 1 news, 1 guide, 1 benchmark, 5 prompts.
- SEO: canonical meta, OG/Twitter, sitemap, robots, JSON-LD.

### 6) Legal
Educational purpose only; images belong to their owners; results vary by params/hardware/version.

---

## 待你确认的选项
1. __主题色/风格__：暗色科技 or 极简白。
2. __语言默认__：中文优先，英文镜像；是否需要更多语言？
3. __是否接入订阅__：Buttondown/ConvertKit/其他或先不接入。
4. __是否需要评论功能__：Giscus/Disqus/Utterances 或暂不启用。
5. __首页 Hero 文案基调__：更技术向 or 更媒体向。

确认后我将直接初始化代码仓库，输出可部署的 Next.js + MDX 模板与示例内容。
