# Nano Banana — AI Image Editing Benchmarks & Quickstart

> Free AI image editing benchmarks, prompt recipes, and quickstart guides for the Nano Banana model. Compare with FLUX Kontext and other models using reproducible test cases.

🌐 **Live Site**: [nano-banana.live](https://www.nano-banana.live) | [English](#english) | [中文](#中文)

---

## English

### 🎯 Project Overview
**Nano Banana** is a comprehensive resource hub for the emerging AI image editing model. This site provides reproducible benchmarks, prompt recipes, and quickstart guides to help users evaluate and master text-guided image editing.

**Target Audience**: AI artists, designers, AIGC enthusiasts, prompt engineers, and media professionals.

**Key Features**:
- **Reproducible Benchmarks**: Fixed inputs, parameters, and pipelines for fair model comparisons
- **Prompt Library**: Copy-ready recipes with optimized parameters for consistent results  
- **5-Minute Quickstart**: Get your first edit running in minutes (Colab + Local routes)
- **A/B Comparisons**: Interactive sliders to compare results side-by-side

### 🌐 Domain & Branding
- **Primary**: `nano-banana.live` (emphasizing "live" updates)
- **Secondary**: `nano-banana.online` (backup/mirror)
- **Brand**: Nano Banana

### 📋 Site Architecture
- **Home `/`**: Model intro + latest updates + benchmark previews + quickstart entry
- **Benchmarks `/benchmarks`**: Reproducible test cases with downloadable inputs/outputs
  - `/benchmarks/portrait-editing`: Portrait editing comparison (Nano Banana vs FLUX Kontext)
- **Prompts `/prompts`**: Categorized prompt recipes with copy buttons and parameters
- **Quickstart `/guides/quickstart`**: Two routes - Colab (zero setup) or Local (full control)
- **FAQ `/faq`**: Compatibility, VRAM requirements, troubleshooting, ethics
- **About `/about`**: Site info, disclaimers, contact
- **Legal**: Privacy Policy `/privacy` and Terms of Service `/terms`

### 🛠 Tech Stack
- **Framework**: Static HTML/CSS (current), Next.js + MDX (planned)
- **Deployment**: Vercel/Netlify with CDN
- **Features**: A/B comparison sliders, lightbox galleries, copy-to-clipboard prompts
- **SEO**: Canonical URLs, OG/Twitter cards, JSON-LD structured data, sitemap

### 🔍 SEO & Performance
- **URL Structure**: Clean, lowercase paths (`/guides/quickstart`, `/benchmarks/portrait-editing`)
- **Meta Data**: Consistent title patterns, descriptions, OG/Twitter cards
- **Structured Data**: JSON-LD for articles, benchmarks, and galleries
- **Performance**: Optimized images, minimal CSS/JS, fast loading
- **Sitemap**: Auto-generated `/sitemap.xml` with all pages
- **Robots**: SEO-friendly `/robots.txt`

### 📈 Latest Updates (2025-08-15)

#### Navigation Enhancement & Brand Strengthening
1. **Navigation Layout Redesign**:
   - Implemented split layout: "Nano Banana" brand left-aligned, other nav items right-aligned
   - Used `justify-content: space-between` and `.nav-links` container for perfect left-right distribution
   - Removed redundant "Image Editing" badges from all pages for cleaner navigation

2. **Brand Identity Enhancement**:
   - Added favicon icons to "Nano Banana" links across all 11 pages (20x20px)
   - Enhanced brand link styling: bold font, full opacity, optimized icon-text spacing
   - Ensured entire brand area is clickable for better UX

3. **Technical Implementation**:
   - CSS: `.nav a:first-child` uses flexbox layout with `gap: 8px` for icon-text spacing
   - HTML: Consistent favicon usage `<img src="./favicon.png">` (root) or `../favicon.png` (subdirs)
   - Link fixes: Homepage uses `./index.html` to avoid local file:// protocol issues

4. **Coverage**:
   - ✅ All 11 HTML pages updated
   - ✅ Unified navigation structure
   - ✅ Desktop and mobile responsive
   - ✅ Local development and production compatible

---

## 中文

### 🎯 项目概述
**Nano Banana** 是新兴AI图像编辑模型的综合资源中心。本站提供可复现的基准测试、提示词配方和快速入门指南，帮助用户评估和掌握文本引导的图像编辑技术。

**目标用户**: AI艺术家、设计师、AIGC爱好者、提示词工程师和媒体从业者。

**核心功能**:
- **可复现基准**: 固定输入、参数和流程，确保模型对比的公平性
- **提示词库**: 可直接复制的配方，附带优化参数以获得一致结果
- **5分钟快速入门**: 快速开始你的第一次编辑（Colab + 本地两种方式）
- **A/B对比**: 交互式滑块并排比较结果

### 🌐 域名与品牌
- **主域名**: `nano-banana.live`（强调"实时"更新）
- **备用域名**: `nano-banana.online`（备份/镜像）
- **品牌**: Nano Banana

### 📋 网站架构
- **首页 `/`**: 模型介绍 + 最新更新 + 基准预览 + 快速入门入口
- **基准测试 `/benchmarks`**: 可复现测试案例，提供可下载的输入/输出
  - `/benchmarks/portrait-editing`: 肖像编辑对比（Nano Banana vs FLUX Kontext）
- **提示词 `/prompts`**: 分类提示词配方，带复制按钮和参数
- **快速入门 `/guides/quickstart`**: 两种路线 - Colab（零配置）或本地（完全控制）
- **常见问题 `/faq`**: 兼容性、显存需求、故障排除、伦理
- **关于 `/about`**: 网站信息、免责声明、联系方式
- **法律**: 隐私政策 `/privacy` 和服务条款 `/terms`

### 🛠 技术栈
- **框架**: 静态HTML/CSS（当前），Next.js + MDX（计划中）
- **部署**: Vercel/Netlify with CDN
- **功能**: A/B对比滑块、灯箱画廊、提示词一键复制
- **SEO**: 规范URL、OG/Twitter卡片、JSON-LD结构化数据、站点地图

### 🔍 SEO与性能
- **URL结构**: 简洁小写路径（`/guides/quickstart`、`/benchmarks/portrait-editing`）
- **元数据**: 一致的标题模式、描述、OG/Twitter卡片
- **结构化数据**: 文章、基准和画廊的JSON-LD
- **性能**: 优化图片、精简CSS/JS、快速加载
- **站点地图**: 自动生成的 `/sitemap.xml` 包含所有页面
- **Robots**: SEO友好的 `/robots.txt`

### 📈 最新更新 (2025-08-15)

#### 导航增强与品牌强化
1. **导航布局重新设计**:
   - 实现分离式布局："Nano Banana"品牌左对齐，其他导航项右对齐
   - 使用 `justify-content: space-between` 和 `.nav-links` 容器实现完美的左右分布
   - 移除所有页面的冗余"Image Editing"标签，导航更简洁

2. **品牌标识增强**:
   - 在所有11个页面的"Nano Banana"链接前添加favicon图标（20x20px）
   - 增强品牌链接样式：粗体字体、完全不透明度、优化图标文字间距
   - 确保整个品牌区域可点击，提升用户体验

3. **技术实现**:
   - CSS: `.nav a:first-child` 使用flexbox布局，`gap: 8px` 实现图标文字间距
   - HTML: 一致使用favicon `<img src="./favicon.png">`（根目录）或 `../favicon.png`（子目录）
   - 链接修复：首页使用 `./index.html` 避免本地 file:// 协议问题

4. **覆盖范围**:
   - ✅ 所有11个HTML页面已更新
   - ✅ 统一导航结构
   - ✅ 桌面端和移动端响应式
   - ✅ 本地开发和生产环境兼容

### 🚀 部署说明

#### 域名重定向策略
- **目标**: 统一收敛到唯一规范域名 `https://www.nano-banana.live`
- **配置**: 已在 `vercel.json` 中设置301重定向
- **效果**: 所有域名访问（裸域、.online等）都会重定向到主域名

#### 快速部署到Vercel
1. 推送代码到Git仓库
2. Vercel → New Project → Import 该仓库
3. Root Directory 选择 `web/`，Framework 选择 "Other"
4. Build Command 和 Output Directory 留空（静态站点）
5. 在Vercel Domains中绑定域名，配置DNS记录

## SEO & GSC 诊断记录（2025-08-15）
- 统一样式表引用与相对路径；移除所有 `?v=` 版本参数，避免缓存与渲染偏差（参考 `web/*.html`）。
- 全站页面已覆盖 canonical（404 除外）：`/`、`/benchmarks/`、`/prompts/`、`/guides/quickstart.html`、`/faq.html`、`/about.html`、`/blog.html`、`/privacy.html`、`/terms.html`、`/benchmarks/portrait-editing.html`。
  - `web/sitemap.xml` 规范化为目录形式：`/benchmarks/`、`/prompts/`，与对应页面 canonical 一致。
  - `vercel.json` 强制 301 收敛到主域 `https://www.nano-banana.live`；`web/robots.txt` 允许抓取并指向站点地图。
  - 新增：`web/404.html` 添加 `<meta name="robots" content="noindex, nofollow">`，防止 404 被索引。

### 上午追加更新（2025-08-15）
 - `web/sitemap.xml`：将所有 URL 的 `<lastmod>` 统一更新为 `2025-08-15`；保持 `/benchmarks/`、`/prompts/` 目录规范与页面 canonical 一致。
 - `web/robots.txt`：重构为显式分组，确保被各爬虫正确解析。
   - 保留通用组 `User-agent: *`：`Allow: /`，`Disallow: /admin/`、`/private/`。
   - 新增并允许主流与 AI 爬虫（继续屏蔽 /admin/ 与 /private/）：
     - OpenAI：`GPTBot`
     - Anthropic：`Claude-Web`、`Anthropic-AI`
     - Perplexity：`PerplexityBot`
     - Google：`Googlebot`、`GoogleOther`、`Google-Extended`
     - Apple：`Applebot`、`Applebot-Extended`
     - Meta：`Meta-ExternalAgent`
     - Amazon：`Amazonbot`
     - Common Crawl：`CCBot`
     - ByteDance：`Bytespider`
     - DuckDuckGo：`DuckDuckBot`
     - Yahoo：`Slurp`
     - Yandex：`YandexBot`
     - 韩国：`Yeti (Naver)`、`NaverBot`
   - 保留 LLM 内容说明注释：`/llms.txt`、`/llms-full.txt`。
 - 部署与验证建议：
   - 部署后（如使用 Cloudflare），建议对 `/sitemap.xml` 执行缓存清除。
   - 在线核对：`/sitemap.xml` 的 `<lastmod>` 是否均为 `2025-08-15`；`/robots.txt` 是否出现新分组。
   - 在 GSC 重新提交 `sitemap.xml`；关键页用 “Test Live URL” + “Request Indexing”。

### 在 Google Search Console 的操作建议
1. 确认属性：选择 `https://www.nano-banana.live/`。
2. URL Inspection → Test Live URL（实时测试）：`/`、`/benchmarks/`、`/prompts/`、`/benchmarks/portrait-editing.html`。
3. 若渲染正常，点击 Request Indexing（请求编入索引）。
4. 在 Sitemaps 再次提交 `https://www.nano-banana.live/sitemap.xml`。
5. 观察 24–72 小时，“Pages” 报告是否消除渲染/规范化误报。

### 线上自检（可选）
- 重定向与状态码：
  - `curl -I -L https://nano-banana.live/`
  - `curl -I -L https://nano-banana.online/`
  - `curl -I -L https://www.nano-banana.online/`
- 关键资源：
  - `curl -I https://www.nano-banana.live/styles.css`
  - `curl -I https://www.nano-banana.live/benchmarks/`
  - `curl -I https://www.nano-banana.live/prompts/`
  - `curl -I https://www.nano-banana.live/guides/quickstart.html`

注：GSC 的渲染异常多由旧快照导致。以上步骤可快速让 GSC 获取最新渲染结果。

### 📝 开发说明

#### Favicon生成命令 (ImageMagick)
```bat
magick web\favicon.png -resize 32x32 -strip -define png:compression-level=9 web\favicon-32x32.png
magick web\favicon.png -resize 16x16 -strip -define png:compression-level=9 web\favicon-16x16.png
magick web\favicon.png -define icon:auto-resize=16,32,48 web\favicon.ico
```

#### 项目结构
```
web/
├── index.html              # 首页
├── benchmarks/
│   ├── index.html         # 基准测试列表
│   └── portrait-editing.html  # 肖像编辑对比
├── guides/
│   └── quickstart.html    # 快速入门指南
├── prompts/
│   └── index.html         # 提示词库
├── faq.html               # 常见问题
├── about.html             # 关于页面
├── blog.html              # 博客
├── privacy.html           # 隐私政策
├── terms.html             # 服务条款
├── styles.css             # 全局样式
├── sitemap.xml            # 站点地图
├── robots.txt             # 搜索引擎指令
└── assets/                # 图片资源
```

#### SEO优化清单
- ✅ 所有页面统一标题格式 `| Mzu`
- ✅ 完整的OG/Twitter卡片
- ✅ JSON-LD结构化数据
- ✅ 多尺寸favicon支持
- ✅ 站点地图和robots.txt
- ✅ 规范URL和301重定向
- ✅ 移动端响应式设计
- ✅ 页面加载性能优化

---
