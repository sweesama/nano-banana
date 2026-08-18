# Nano Banana API & Local Alternatives

[Live site](https://www.nano-banana.live/) · [Quickstart](https://www.nano-banana.live/guides/quickstart.html) · [Benchmarks](https://www.nano-banana.live/benchmarks/) · [Prompts](https://www.nano-banana.live/prompts/)

An independent reference site that separates two often-confused workflows:

1. **Nano Banana 2 API:** Google's `gemini-3.1-flash-image` runs in Google's cloud. The Python client runs on your computer, but inference does not.
2. **True local inference:** separate open-weight models such as Qwen-Image-Edit-2511, HiDream-O1-Image, and HunyuanImage 3.0 run on hardware you control.

This repository is not an official Google, Gemini, Qwen, Tencent, or HiDream project.

## Site structure

- `web/index.html` — cloud-vs-local explanation and main routes
- `web/guides/quickstart.html` — model status, official links, and route selection
- `web/blog/` — source-backed tutorials
- `web/benchmarks/` — first-party comparisons and dated third-party snapshots
- `web/prompts/` — copy-ready editing prompts
- `code/nano_api.py` — minimal Google Gemini image-editing client
- `scripts/validate-site.js` — structural and factual regression checks

## Content rules

- Prefer primary sources: provider docs, official repositories, and model cards.
- Never describe a local API client as local inference.
- Date volatile claims such as rankings, pricing, model availability, and hardware guidance.
- Do not invent testimonials, first-hand testing, or precise hardware requirements.
- Automated articles publish only after required sources are readable, a separate source audit passes, and the full static site passes structural, link, metadata, and queue checks.

## Automation

- `weekly-blog.yml` runs three times weekly and publishes one article only after source, fact, and site validation gates pass.
- `sync-benchmarks.yml` runs weekly, validates the static site, and commits only changed benchmark data.
- Both publishers share one concurrency group, rebase on the latest `main`, revalidate, and only then push.
- Run the local checks with:

```bash
node scripts/validate-site.js
python -m py_compile code/nano_api.py
```

## Primary references

- [Google Gemini image generation](https://ai.google.dev/gemini-api/docs/image-generation)
- [Qwen-Image-Edit-2511 model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [HunyuanImage 3.0 repository](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)
- [HiDream-O1-Image repository](https://github.com/HiDream-ai/HiDream-O1-Image)

## 中文说明

本项目是一个独立资料站，不是 Google 官方网站。Nano Banana 2 是 Google 托管的 Gemini 3.1 Flash Image：你可以在本地运行 Python 客户端，但模型推理仍发生在云端。真正离线运行需要选择另外的开源权重模型，并以各项目的官方仓库、许可证和硬件说明为准。
