# Nano Banana — AI Image Editing (Client & Offline)

> The definitive resource hub for Nano Banana. Featuring the **Pro Client** (Google Gemini 3.x Image) and **Open-Source Local Alternatives** such as Z-Image, Qwen-Image, and GLM-Image.
> *Looking for the original model docs? Check the [Legacy Docs folder](./legacy_docs/).*

🌐 **Live Site**: [nano-banana.live](https://www.nano-banana.live) | [English](#english) | [中文](#中文)

---

## English

### 🎯 Project Overview
**Nano Banana** cuts through the confusion of AI image editing. We provide the truth about "running locally" vs "cloud inference."
This site hosts the official **Nano Banana Client** documentation, reproducible benchmarks, and prompt recipes for the latest Google Gemini image stack and strong open-source local alternatives.

**Key Features**:
- **The "Truth" about Local Run**: Clear distinction between the Python Client (CPU-friendly) and Offline Models (GPU-heavy).
- **Pro Client Guide**: How to control Gemini image models from your terminal with `google-genai`.
- **Prompt Recipes**: Instruction-based prompts optimized for modern instruction-tuned models.
- **Reproducible Benchmarks**: Side-by-side comparisons with FLUX and SDXL.

### 🌐 Domain & Branding
- **Primary**: `nano-banana.live`
- **Secondary**: `nano-banana.online` (redirects to live)

### 📋 Site Architecture
- **Home `/`**: Intro + Quickstart Entry
- **Guides `/guides/quickstart`**:
  - **Route A (Pro)**: Python Client (Requires API Key, Runs on CPU).
  - **Route B (Offline)**: Z-Image / Qwen-Image / GLM-Image (GPU requirements vary by model).
- **Blog & Tutorials `/blog`**:
  - `/blog/how-to-run-locally.html`: The definitive guide to the Client.
  - `/blog/hardware-requirements.html`: VRAM Analysis (Client vs Offline).
  - `/blog/troubleshooting-install.html`: Fixing API/Proxy issues.
  - `/blog/top-prompt-recipes.html`: Gemini instruction recipes.
- **Benchmarks `/benchmarks`**: Fair comparisons.
- **Prompts `/prompts`**: Copy-paste JSON/Python dictionaries.
- **FAQ `/faq`**: Address "Closed Source" reality and licensing.

### 🛠 Tech Stack
- **Web**: Pure Static HTML5/CSS3 (Glassmorphism Design).
- **Model (Pro)**: Google Gemini 3.1 Flash Image / Gemini 3 Pro Image (Cloud).
- **Model (Offline)**: Z-Image / Qwen-Image / GLM-Image / HunyuanImage 3.0 / SDXL Turbo (Local via Diffusers and related serving stacks).

### 📈 Latest Updates (2026-04-08)

#### Quick Start Page — Major Content Refresh
- **Nano Banana 2**: Replaced Nano Banana Pro with the newer Nano Banana 2 (Gemini 3.1 Flash Image) across the entire page. AA T2I #2 / Edit #3.
- **HunyuanImage 3.0 Instruct**: Added as the 4th open-source model. 80B MoE, CoT reasoning, multi-image fusion. AA open-weight Edit #1 🏆.
- **AA Rankings Updated**: All Artificial Analysis rankings refreshed to April 2026 data. Qwen-Image now open-weight T2I #2.
- **SEO Meta**: Updated description and keywords to include HunyuanImage and "self hosted".
- **Sitemap**: Updated quickstart.html lastmod to 2026-04-08.

---

### 📈 Previous Updates (2026-02-01)

#### Z-Image Page Enhancements
- **Gallery Carousels**: Refined image selection across all capability sections (removed inconsistent aspect ratios).
- **Architecture Diagram**: Enlarged to full width for better readability.
- **Internal Navigation**: All links now properly route to internal pages.

#### Quick Start Page Updates
- **Closed-Source Warning**: Added prominent notice that Nano Banana Pro cannot run locally.
- **Z-Image Section**: Added 6-image CDN showcase and "Read Full Guide" button linking to z-image.html.
- **External Links**: Hugging Face and ModelScope links now open in new tabs.

---

### 📈 Previous Updates (2026-01-20)

#### 1. Content Strategy Pivot: "The Truth"
- **New Blog Section**: Launched 4 deep-dive articles addressing the "Local Run" confusion.
- **GSC Alignment**: Content now directly targets search queries like "nano banana local install" and "requirements".
- **FAQ Overhaul**: Rewrote FAQ to explicitly state that Nano Banana Pro is closed-source, while offering Z-Image as the offline alternative.

#### 2. Visual & UX Upgrades
- **Glassmorphism UI**: Updated Blog and FAQ with premium glass cards and gradients.
- **Accordion FAQ**: Replaced dense grids with expandable details/summary animations.
- **No Underlines**: Cleaned up link styling across article cards.

#### 3. SEO Improvements
- **Sitemap**: Updates `2026-01-20` for all 14 pages.
- **Favicons**: Unified branding across root/subdirectories.
- **Canonical URLs**: Full coverage for all new blog posts.

---

## 中文 (Chinese)

### 🎯 项目概述
**Nano Banana** 是 AI 图像编辑领域的"真相"中心。我们致力于澄清"本地运行"与"云端推理"的区别。
本站提供 **Nano Banana Client**（Google Gemini 3.x Image）以及 **Z-Image / Qwen-Image / GLM-Image** 等离线替代方案的文档、基准测试和 Prompt 配方。

**核心功能**:
- **理清"本地运行"**: 明确区分 Python 客户端 (任意电脑) 与 离线大模型 (需要高端显卡)。
- **Pro 客户端指南**: 如何在终端中通过 `google-genai` API 操控云端图像模型。
- **Prompt 配方**: 专为指令遵循模型优化的自然语言 Prompt。
- **基准测试**: 与 FLUX 和 SDXL 的公平对比。

### 📋 网站架构
- **首页 `/`**: 入口与概览
- **快速入门 `/guides/quickstart`**:
  - **路线 A (Pro)**: Python Client (需要 API Key, CPU 即可运行).
  - **路线 B (离线)**: Z-Image / Qwen-Image / GLM-Image（显卡要求因模型而异）.
- **博客 `/blog`**:
  - `/blog/how-to-run-locally.html`: 客户端运行权威指南.
  - `/blog/hardware-requirements.html`: 硬件配置分析.
  - `/blog/troubleshooting-install.html`: 解决 API/代理报错.
  - `/blog/top-prompt-recipes.html`: Gemini 专用 Prompt.
- **常见问题 `/faq`**: 明确闭源/开源界限。

### 📈 最新更新 (2026-01-20)

#### 1. 内容战略转型
- **博客上线**: 4篇核心技术文章，精准回应 "Local Install" 搜索意图。
- **FAQ 重写**: 诚实说明 Pro 版的闭源属性，并指引离线用户去使用 Z-Image。

#### 2. 视觉升级
- **设计风格**: 博客和 FAQ 全面采用毛玻璃 (Glassmorphism) 风格。
- **交互优化**: FAQ 改为折叠面板 (Accordion) 样式，博客卡片移除下划线。

#### 3. SEO 优化
- **Sitemap 更新**: 全站 14 个页面时间戳更新为 2026-01-20。
