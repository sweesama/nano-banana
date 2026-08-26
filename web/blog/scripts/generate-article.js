import OpenAI from 'openai';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const BLOG_DIR = path.resolve(__dirname, '..');
const WEB_DIR = path.resolve(BLOG_DIR, '..');
const ROOT_DIR = path.resolve(WEB_DIR, '..');
const QUEUE_PATH = path.join(BLOG_DIR, 'queue.json');
const ARTICLES_PATH = path.join(BLOG_DIR, 'articles.json');
const SEO_RESEARCH_PATH = path.join(BLOG_DIR, 'seo-research.json');
const BLOG_INDEX_PATH = path.join(WEB_DIR, 'blog.html');
const SITEMAP_PATH = path.join(WEB_DIR, 'sitemap.xml');
const API_KEY = process.env.NVIDIA_API_KEY;

function parseModelList(value, fallback) {
  const parsed = String(value || '').split(',').map(item => item.trim()).filter(Boolean);
  return parsed.length > 0 ? [...new Set(parsed)] : fallback;
}

// NVIDIA hosted free endpoints are prototype services and may rotate or throttle.
// These defaults were re-verified in NVIDIA's official catalog on 2026-08-26.
const MODELS = parseModelList(process.env.BLOG_MODEL_LIST, [
  'deepseek-ai/deepseek-v4-flash-0731',
  'nvidia/nemotron-3.5-lightning-30b-a3b',
  'stepfun-ai/step-3.7-flash',
  'nvidia/nemotron-3-ultra-550b-a55b',
]);
const VERIFIER_MODELS = parseModelList(process.env.BLOG_VERIFIER_MODEL_LIST, [
  'nvidia/nemotron-3.5-lightning-30b-a3b',
  'stepfun-ai/step-3.7-flash',
  'deepseek-ai/deepseek-v4-flash-0731',
]);
const API_TIMEOUT_MS = Number(process.env.BLOG_API_TIMEOUT_MS || 180000);
const MAX_MODEL_ATTEMPTS = 2;
const disabledModels = new Set();
const DEPTHS = { brief: [450, 700], standard: [700, 1100], deep: [1100, 1600] };
const BANNED_PHRASES = ['in today\'s rapidly evolving', 'game-changer', 'seamlessly', 'revolutionize', 'it is worth noting', 'delve into', 'unlock the power'];
const ALLOWED_TAGS = new Set(['p', 'h2', 'h3', 'ul', 'ol', 'li', 'pre', 'code', 'strong', 'em', 'blockquote', 'a', 'br']);
const AUTHORITATIVE_SOURCE_HOSTS = new Set([
  'ai.google.dev',
  'blog.google',
  'developers.google.com',
  'developers.openai.com',
  'github.com',
  'huggingface.co',
  'opensource.org',
  'artificialanalysis.ai',
  'www.runpod.io',
  'runpod.io',
  'docs.runpod.io',
  'vast.ai',
  'www.vast.ai',
]);
const SITE_HOST = 'www.nano-banana.live';

const ai = API_KEY ? new OpenAI({ apiKey: API_KEY, baseURL: 'https://integrate.api.nvidia.com/v1' }) : null;

function modelsForItem(item) {
  return MODELS;
}

function escapeHtml(value) {
  return String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function sourceHost(url) {
  try {
    return new URL(url).hostname.toLowerCase();
  } catch {
    return '';
  }
}

function isAuthoritativeExternalSource(url) {
  const host = sourceHost(url);
  return host !== SITE_HOST && AUTHORITATIVE_SOURCE_HOSTS.has(host);
}

function findSeoCluster(item, research) {
  const cluster = research.clusters.find(entry => entry.categories.includes(item.category));
  if (!cluster) throw new Error(`No SEO research cluster matches category: ${item.category}`);
  return cluster;
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function parseJson(text) {
  const cleaned = text.replace(/^```json\s*/i, '').replace(/\s*```$/i, '').trim();
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start < 0 || end < start) throw new Error('Model did not return a JSON object.');
  return JSON.parse(cleaned.slice(start, end + 1));
}

function getErrorStatus(error) {
  return Number(error?.status || error?.response?.status || 0);
}

function isJsonModeUnsupported(error) {
  return getErrorStatus(error) === 400 && /response[_ -]?format|json[_ -]?object|structured output/i.test(error?.message || '');
}

function classifyModelError(error) {
  const status = getErrorStatus(error);
  const message = error?.message || '';
  if (status === 401 || status === 403) return 'auth';
  if (status === 404 || status === 410 || status === 400) return 'permanent';
  if (status === 429 || status >= 500 || /quota|RESOURCE_EXHAUSTED|high demand/i.test(message)) return 'transient';
  if (/AbortError|aborted|timeout/i.test(message)) return 'timeout';
  if (/Connection|ECONNRESET|socket|network|fetch/i.test(message)) return 'transient';
  if (/JSON object|JSON.parse|Unexpected token|Unexpected end/i.test(message)) return 'retryable-output';
  return 'unknown';
}

function normalizeDescription(value, maxLength = 170) {
  const compact = String(value || '').replace(/\s+/g, ' ').trim();
  if (compact.length <= maxLength) return compact;
  const candidate = compact.slice(0, maxLength - 1);
  const sentenceBoundary = Math.max(candidate.lastIndexOf('. '), candidate.lastIndexOf('! '), candidate.lastIndexOf('? '));
  const wordBoundary = candidate.lastIndexOf(' ');
  const cutAt = sentenceBoundary >= 100 ? sentenceBoundary + 1 : wordBoundary;
  const shortened = candidate.slice(0, cutAt > 0 ? cutAt : maxLength - 1).replace(/[\s,;:\-]+$/g, '');
  return /[.!?]$/.test(shortened) ? shortened : `${shortened}.`;
}

function normalizeTitle(value, requiredTerm, maxLength = 70) {
  const compact = String(value || '').replace(/\s+/g, ' ').trim();
  if (compact.length <= maxLength) return compact;
  const truncate = text => {
    const candidate = text.slice(0, maxLength + 1);
    const wordBoundary = candidate.lastIndexOf(' ');
    return candidate.slice(0, wordBoundary >= 20 ? wordBoundary : maxLength).replace(/[\s:;,.\-]+$/g, '');
  };
  const shortened = truncate(compact);
  if (!requiredTerm || shortened.toLowerCase().includes(requiredTerm.toLowerCase())) return shortened;

  const termPattern = new RegExp(requiredTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'ig');
  const remainder = compact.replace(termPattern, '').replace(/^[\s:;,.\-]+|[\s:;,.\-]+$/g, '');
  return truncate(`${requiredTerm}: ${remainder}`);
}

async function requestJsonModel(model, messages, { label, temperature, maxTokens }) {
  if (disabledModels.has(model)) throw new Error(`Model ${model} is disabled for this run.`);
  let lastError;
  for (let attempt = 1; attempt <= MAX_MODEL_ATTEMPTS; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(new Error(`Model timeout after ${Math.round(API_TIMEOUT_MS / 1000)}s`)), API_TIMEOUT_MS);
    const payload = {
      model,
      temperature,
      top_p: 0.95,
      max_tokens: maxTokens,
      response_format: { type: 'json_object' },
      messages,
    };
    try {
      console.log(`[${label}] Trying ${model} (${attempt}/${MAX_MODEL_ATTEMPTS})...`);
      let response;
      try {
        response = await ai.chat.completions.create(payload, { signal: controller.signal });
      } catch (error) {
        if (!isJsonModeUnsupported(error)) throw error;
        console.log(`[${label}] ${model} does not accept response_format; retrying with prompt-enforced JSON.`);
        response = await ai.chat.completions.create({ ...payload, response_format: undefined }, { signal: controller.signal });
      }
      const data = parseJson(response.choices?.[0]?.message?.content || '');
      console.log(`[${label}] ${model} returned valid JSON.`);
      return data;
    } catch (error) {
      lastError = error;
      const kind = classifyModelError(error);
      const status = getErrorStatus(error);
      const summary = status ? `HTTP ${status}` : (error.message || 'Unknown error');
      console.warn(`[${label}] ${model} failed: ${summary}`);
      if (kind === 'auth') throw new Error(`NVIDIA API authentication failed: ${summary}`);
      if (kind === 'permanent') {
        disabledModels.add(model);
        console.warn(`[${label}] ${model} is disabled for the remainder of this run.`);
        break;
      }
      if (kind === 'timeout') {
        disabledModels.add(model);
        console.warn(`[${label}] ${model} timed out and is disabled for the remainder of this run.`);
        break;
      }
      if (attempt < MAX_MODEL_ATTEMPTS && ['transient', 'retryable-output', 'unknown'].includes(kind)) {
        await new Promise(resolve => setTimeout(resolve, kind === 'transient' ? 12000 : 3000));
        continue;
      }
      break;
    } finally {
      clearTimeout(timeout);
    }
  }
  throw lastError || new Error(`Model ${model} failed.`);
}

function sanitizeHtml(input) {
  const codeBlocks = [];
  const protectedInput = String(input || '').replace(/<pre\b[^>]*>\s*<code\b[^>]*>([\s\S]*?)<\/code>\s*<\/pre>/gi, (_full, code) => {
    const safeCode = String(code)
      .replace(/&(?!(?:amp|lt|gt|quot|#39);)/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    const token = `@@NANO_CODE_BLOCK_${codeBlocks.length}@@`;
    codeBlocks.push(`<pre><code>${safeCode}</code></pre>`);
    return token;
  });
  const sanitized = protectedInput
    .replace(/<!--([\s\S]*?)-->/g, '')
    .replace(/<(script|style|iframe|object|embed|form|input|button)[^>]*>[\s\S]*?<\/\1>/gi, '')
    .replace(/<\/?([a-z0-9-]+)([^>]*)>/gi, (full, tag, attrs) => {
      const lower = tag.toLowerCase();
      if (!ALLOWED_TAGS.has(lower)) return '';
      if (full.startsWith('</')) return `</${lower}>`;
      if (lower === 'a') {
        const hrefMatch = attrs.match(/href\s*=\s*["']([^"']+)["']/i);
        const href = hrefMatch ? hrefMatch[1] : '#';
        const safeHref = /^(https?:\/\/|\/|\.\.\/|#)/i.test(href) ? href : '#';
        const external = /^https?:\/\//i.test(safeHref);
        return `<a href="${escapeHtml(safeHref)}"${external ? ' target="_blank" rel="noopener"' : ''}>`;
      }
      if (lower === 'br') return '<br>';
      return `<${lower}>`;
    })
    .trim();
  return sanitized.replace(/@@NANO_CODE_BLOCK_(\d+)@@/g, (_full, index) => codeBlocks[Number(index)] || '');
}

function formatDisplayDate(isoDate) {
  const date = new Date(`${isoDate}T00:00:00Z`);
  return Number.isNaN(date.getTime())
    ? isoDate
    : date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' });
}

function countWords(value) {
  return String(value).replace(/<[^>]+>/g, ' ').trim().split(/\s+/).filter(Boolean).length;
}

function scoreArticle(article, item, cluster) {
  const text = `${article.title} ${article.description} ${article.content}`.toLowerCase();
  const [minWords] = DEPTHS[item.depth] || DEPTHS.standard;
  const primaryTermInTitle = cluster.primaryTerms.some(term => article.title.toLowerCase().includes(term.toLowerCase()));
  const primaryTermInBody = cluster.primaryTerms.some(term => article.content.toLowerCase().includes(term.toLowerCase()));
  const internalLinkCount = (article.content.match(/href=["'](?:\.\.\/|\/|https:\/\/www\.nano-banana\.live)/gi) || []).length;
  let score = 0;
  if (article.title.length >= 35 && article.title.length <= 70) score += 15;
  if (article.description.length >= 100 && article.description.length <= 170) score += 10;
  if (countWords(article.content) >= minWords) score += 20;
  if ((article.content.match(/<h2>/g) || []).length >= 2) score += 15;
  if ((article.content.match(/<a /g) || []).length >= 2) score += 10;
  if ((article.content.match(/<li>/g) || []).length >= 3) score += 10;
  if (primaryTermInTitle) score += 10;
  if (primaryTermInBody) score += 5;
  if (internalLinkCount >= 2) score += 5;
  if (!BANNED_PHRASES.some(phrase => text.includes(phrase))) score += 5;
  return Math.min(score, 100);
}

function validateArticle(article, item, cluster) {
  const required = ['title', 'description', 'content'];
  for (const field of required) {
    if (typeof article[field] !== 'string' || !article[field].trim()) throw new Error(`Missing article field: ${field}`);
  }
  if (article.title.length > 80) throw new Error('Title is too long.');
  if (article.description.length > 180) throw new Error('Description is too long.');
  if (/<(script|iframe|object|embed|form)\b/i.test(article.content)) throw new Error('Unsafe HTML detected.');
  if (/<a\b[^>]*>(?:(?!<\/a>)[\s\S])*<a\b/i.test(article.content)) throw new Error('Nested anchor detected.');
  if (/href=["']#["']/i.test(article.content)) throw new Error('Placeholder link detected.');
  for (const tag of ['a', 'pre', 'code']) {
    const opens = (article.content.match(new RegExp(`<${tag}\\b`, 'gi')) || []).length;
    const closes = (article.content.match(new RegExp(`</${tag}>`, 'gi')) || []).length;
    if (opens !== closes) throw new Error(`Unbalanced <${tag}> tags.`);
  }
  if (!cluster.primaryTerms.some(term => article.title.toLowerCase().includes(term.toLowerCase()))) throw new Error('Title does not contain a researched primary keyword.');
  if (cluster.internalLinks.filter(url => article.content.includes(url)).length < 2) throw new Error('Article does not contain at least two researched internal links.');
  if (item.sourceUrls.filter(url => article.content.includes(url)).length !== item.sourceUrls.length) throw new Error('Article does not cite every required source near the relevant claim.');
  if (!item.sourceUrls.some(isAuthoritativeExternalSource)) throw new Error('Queue item needs at least one approved authoritative external source.');
  if (/\b(?:we tested|our tests show|our customers|users say)\b/i.test(article.content)) throw new Error('Unsubstantiated first-party experience or testimonial claim detected.');
  if (/\b(?:guaranteed|always|never fails|unlimited)\b/i.test(article.content)) throw new Error('Absolute product claim detected.');
  if (/\bfree tier\b/i.test(article.content) && !item.sourceUrls.some(url => url.includes('ai.google.dev/gemini-api/docs/pricing'))) {
    throw new Error('A free-tier claim requires the live Gemini pricing page as a source.');
  }
}

async function fetchSourceNotes(urls) {
  if (!Array.isArray(urls) || urls.length < 2) throw new Error('Every queue item needs at least two source URLs.');
  if (!urls.some(isAuthoritativeExternalSource)) throw new Error('No approved authoritative external source is configured.');

  return Promise.all(urls.map(async (url, index) => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch(url, {
        headers: {
          accept: 'text/html,application/json,text/plain',
          'user-agent': 'NanoBananaSourceCheck/1.0 (+https://www.nano-banana.live/about.html)',
        },
        redirect: 'follow',
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const raw = await response.text();
      const text = raw
        .replace(/<script[\s\S]*?<\/script>/gi, ' ')
        .replace(/<style[\s\S]*?<\/style>/gi, ' ')
        .replace(/<[^>]+>/g, ' ')
        .replace(/&nbsp;|&#160;/gi, ' ')
        .replace(/&amp;/gi, '&')
        .replace(/\s+/g, ' ')
        .trim();
      if (text.length < 200) throw new Error('source text is too short to verify claims');
      return { index: index + 1, url, text: text.slice(0, 9000) };
    } catch (error) {
      throw new Error(`Required source unavailable: ${url} (${error.message})`);
    } finally {
      clearTimeout(timeout);
    }
  }));
}

function promptFor(item, existingArticles, sourceRecords, cluster, requiredTerm) {
  const [minWords, maxWords] = DEPTHS[item.depth] || DEPTHS.standard;
  const existingTitles = existingArticles.map(article => article.title).join(' | ');
  const sourceNotes = sourceRecords.map(source => `[SOURCE ${source.index}] ${source.url}\n${source.text}`).join('\n\n');
  return `You write an accurate English-first technical article for Nano Banana, a site about Gemini image APIs and open-weight local image models.

Topic keyword: ${item.keyword}
Category: ${item.category}
Target length: ${minWords}-${maxWords} words
Required sources: ${item.sourceUrls.join(', ')}
Primary SEO terms: ${cluster.primaryTerms.join(', ')}
Required title phrase (must appear verbatim, case-insensitive, in the title): "${requiredTerm}"
Related SEO terms: ${cluster.relatedTerms.join(', ')}
Search intent: ${cluster.intent}
Preferred internal links: ${cluster.internalLinks.join(', ')}
Source notes fetched at generation time:
${sourceNotes}
Existing article titles to avoid repeating: ${existingTitles}

Return JSON only with exactly these fields:
{"title":"...","description":"...","content":"..."}

Content rules:
- Return article body HTML only, without h1, html, head, body, style, script, or markdown fences.
- Use h2, h3, p, ul, ol, li, pre, code, strong, em, blockquote, and a tags only.
- Include at least two h2 headings, one practical list, and concrete steps or comparisons.
- The title MUST contain the required title phrase above, word-for-word (case-insensitive), and use related terms only where they help the reader.
- Answer the search intent directly in the opening paragraph; never stuff keywords or write a generic introduction.
- Include at least two links from the preferred internal-link list with descriptive anchor text.
- Cite claims by linking to source URLs near the relevant discussion. Do not add a Sources section — the publishing script generates one automatically.
- Treat the supplied source text as the complete evidence boundary. A fact absent from it must be omitted or explicitly labeled unknown.
- Keep volatile prices, quotas, rankings, and availability dated and linked to the source that states them.
- For benchmark or Elo topics, prefer explaining how to interpret the metric. Do not reproduce a live ranking table, exact current ranks, prices, sample counts, confidence intervals, or model scores when fetched sources show different snapshots. Hypothetical numbers must be explicitly labeled as examples.
- A keyword may appear as a question, but answer it directly. Never imply that an API client makes a hosted model local or offline.
- Always close every HTML tag properly. Close <a> tags with </a>, never use <a href="#"> as a closing tag.
- Put code samples inside <pre><code> blocks and HTML-escape literal angle brackets inside code.
- Never invent benchmark scores, model release status, API prices, hardware requirements, or product features. If a source does not confirm a detail, say that it is unknown.
- Clearly distinguish cloud APIs from local open-weight models.
- Mention Nano Banana naturally only when relevant. Do not keyword-stuff.
- Avoid generic AI phrases such as “game-changer”, “seamlessly”, “delve into”, and “unlock the power”.
- Do not claim first-hand testing unless the source or site content explicitly supports it.
- Write for a technically curious beginner and keep the advice actionable.`;
}

async function verifyArticle(article, item, sourceRecords) {
  const evidence = sourceRecords.map(source => `[SOURCE ${source.index}] ${source.url}\n${source.text}`).join('\n\n');
  const messages = [
      {
        role: 'system',
        content: 'You are a strict publication fact checker. Treat the supplied source excerpts as the only evidence for factual product claims. Return JSON only.',
      },
      {
        role: 'user',
        content: `Audit this proposed article before automatic publication.

Topic: ${item.keyword}
Title: ${article.title}
Description: ${article.description}
Article HTML:
${article.content}

Allowed evidence:
${evidence}

Return exactly:
{"verdict":"pass|fail","unsupportedClaims":[],"contradictions":[],"missingQualifications":[]}

Fail when a claim about model identity, capabilities, weights, license, pricing, quota, ranking, release status, hardware, API syntax, or provider policy is absent from or contradicted by the evidence. Fail undated volatile numbers, claims of first-hand testing without evidence, universal superiority claims, and language that confuses a local API client with local inference. General workflow advice does not need a citation.`,
      },
    ];
  let lastError;
  for (const verifierModel of VERIFIER_MODELS) {
    if (disabledModels.has(verifierModel)) continue;
    let audit;
    try {
      audit = await requestJsonModel(verifierModel, messages, {
        label: 'source-audit',
        temperature: 0,
        maxTokens: 1800,
      });
    } catch (error) {
      lastError = error;
      continue;
    }
    const issueCount = ['unsupportedClaims', 'contradictions', 'missingQualifications']
      .reduce((total, key) => total + (Array.isArray(audit[key]) ? audit[key].length : 1), 0);
    if (audit.verdict !== 'pass' || issueCount > 0) {
      const auditError = new Error(`Source audit failed: ${JSON.stringify(audit)}`);
      auditError.audit = audit;
      throw auditError;
    }
    return;
  }
  throw lastError || new Error('All verifier models were unavailable.');
}

function prepareArticle(article, item, cluster, requiredTerm) {
  const originalTitle = article.title;
  const originalDescription = article.description;
  article.title = normalizeTitle(article.title, requiredTerm);
  article.description = normalizeDescription(article.description);
  if (article.title !== originalTitle) {
    console.log(`Shortened title from ${String(originalTitle || '').length} to ${article.title.length} characters.`);
  }
  if (article.description !== originalDescription) {
    console.log(`Shortened meta description from ${String(originalDescription || '').length} to ${article.description.length} characters.`);
  }
  article.content = sanitizeHtml(article.content);
  validateArticle(article, item, cluster);
  return article;
}

async function repairArticleAfterAudit(article, item, sourceRecords, cluster, requiredTerm, audit) {
  const evidence = sourceRecords.map(source => `[SOURCE ${source.index}] ${source.url}\n${source.text}`).join('\n\n');
  const messages = [
    {
      role: 'system',
      content: 'You are a conservative publication editor. Revise an article to resolve every fact-check issue using only the supplied evidence. Return JSON only.',
    },
    {
      role: 'user',
      content: `Revise this article after a failed source audit.

Topic: ${item.keyword}
Required title phrase: ${requiredTerm}
Required source URLs: ${item.sourceUrls.join(', ')}
Required internal links: ${cluster.internalLinks.join(', ')}
Audit findings: ${JSON.stringify(audit)}

Current article:
${JSON.stringify(article)}

Allowed evidence:
${evidence}

Return exactly {"title":"...","description":"...","content":"..."}.
Resolve every audit item; do not merely add disclaimers around contradicted claims. Remove unsupported or conflicting ranks, scores, prices, sample counts, open-weight labels, and model status claims. For benchmark topics, explain the interpretation method and snapshot drift without copying a current leaderboard table. Preserve every required source URL and at least two required internal links in relevant anchor tags. Keep the title phrase verbatim. Use valid article-body HTML only.`,
    },
  ];

  let lastError;
  const repairModels = MODELS.filter(model => !disabledModels.has(model));
  for (const repairModel of repairModels) {
    if (disabledModels.has(repairModel)) continue;
    try {
      console.log(`[source-repair] Revising with ${repairModel}...`);
      const repaired = await requestJsonModel(repairModel, messages, {
        label: 'source-repair',
        temperature: 0.15,
        maxTokens: 7000,
      });
      prepareArticle(repaired, item, cluster, requiredTerm);
      await verifyArticle(repaired, item, sourceRecords);
      console.log(`[source-repair] ${repairModel} passed the second source audit.`);
      return repaired;
    } catch (error) {
      lastError = error;
      console.warn(`[source-repair] ${repairModel} revision failed: ${error.message}`);
    }
  }
  throw lastError || new Error('All configured models failed to repair the source audit findings.');
}

async function generateArticle(item, existingArticles, cluster) {
  const sourceRecords = await fetchSourceNotes(item.sourceUrls);
  const requiredTerm = cluster.primaryTerms[0];
  let lastError;
  for (const model of modelsForItem(item)) {
    if (disabledModels.has(model)) continue;
    let article;
    try {
      article = await requestJsonModel(model, [
          { role: 'system', content: 'You produce precise, source-aware technical HTML articles. Output valid JSON only.' },
          { role: 'user', content: promptFor(item, existingArticles, sourceRecords, cluster, requiredTerm) }
        ], {
        label: 'article',
        temperature: 0.45,
        maxTokens: 7000,
      });
    } catch (error) {
      lastError = error;
      console.warn(`Model ${model} failed to respond: ${error.message}`);
      continue;
    }
    try {
      prepareArticle(article, item, cluster, requiredTerm);
      try {
        await verifyArticle(article, item, sourceRecords);
      } catch (error) {
        if (!error.audit) throw error;
        // 排行榜快照最容易发生数据漂移；与其反复修补数字，不如换模型重新生成一篇不依赖易变数字的文章。
        if (item.category === 'Benchmarks') throw error;
        article = await repairArticleAfterAudit(article, item, sourceRecords, cluster, requiredTerm, error.audit);
      }
    } catch (error) {
      lastError = error;
      console.warn(`Model ${model} produced invalid content: ${error.message}`);
      continue;
    }
    return article;
  }
  throw lastError || new Error('All configured models failed to produce a valid article.');
}

function buildArticleHtml(article, item, date) {
  const sources = item.sourceUrls.map(url => `<li><a href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(url)}</a></li>`).join('');
  const schema = JSON.stringify({
    '@context': 'https://schema.org',
    '@type': 'Article',
    headline: article.title,
    description: article.description,
    datePublished: date,
    dateModified: date,
    author: { '@type': 'Organization', name: 'Nano Banana' },
    publisher: { '@type': 'Organization', name: 'Nano Banana' },
    mainEntityOfPage: `https://www.nano-banana.live/blog/${item.slug}.html`,
    isPartOf: { '@type': 'Blog', name: 'Nano Banana Blog', url: 'https://www.nano-banana.live/blog.html' },
  }).replace(/</g, '\\u003c');
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>${escapeHtml(article.title)} | Nano Banana</title>
  <meta name="description" content="${escapeHtml(article.description)}">
  <meta name="robots" content="index, follow">
  <link rel="canonical" href="https://www.nano-banana.live/blog/${escapeHtml(item.slug)}.html">
  <script type="application/ld+json">${schema}</script>
  <link rel="stylesheet" href="../styles.css">
  <script src="../analytics.js" defer></script>
</head>
<body>
  <div class="container">
    <nav class="nav">
      <a href="../index.html"><img src="../favicon.png" alt="Nano Banana" width="20" height="20">Nano Banana</a>
      <div class="nav-links">
        <a href="../benchmarks/index.html">Benchmarks</a>
        <a href="../prompts/index.html">Prompts</a>
        <a href="../guides/quickstart.html">Quickstart</a>
        <a href="../faq.html">FAQ</a>
        <a href="../blog.html" class="active">Blog</a>
      </div>
    </nav>
    <article class="article-container">
      <a href="../blog.html" class="article-back">← Back to Blog</a>
      <header class="article-header">
        <div class="meta" style="justify-content:center;display:flex;">${escapeHtml(item.category)} • ${escapeHtml(formatDisplayDate(date))}</div>
        <h1>${escapeHtml(article.title)}</h1>
        <p class="lead" style="margin:20px auto;">${escapeHtml(article.description)}</p>
      </header>
      <div class="article-content">
        ${article.content}
        <h2>Sources</h2>
        <ul>${sources}</ul>
      </div>
    </article>
    <footer class="footer"><p style="text-align:center;">© <script>document.write(new Date().getFullYear())</script> Nano Banana</p></footer>
  </div>
</body>
</html>
`;
}

function updateBlogIndex(article, item, date) {
  const html = fs.readFileSync(BLOG_INDEX_PATH, 'utf8').replace(/\r\n/g, '\n');
  const gridStart = html.indexOf('<div class="grid"');
  if (gridStart === -1) throw new Error('Blog index grid container not found.');
  const rowEnd = html.indexOf('>', gridStart) + 1;
  const card = `\n        <a href="./blog/${escapeHtml(item.slug)}.html" class="card article-card"><div class="article-image bg-gradient-4" style="display:flex;align-items:center;justify-content:center;"><span style="font-size:40px;">${escapeHtml(item.emoji)}</span></div><div class="article-meta">${escapeHtml(item.category)} • ${escapeHtml(formatDisplayDate(date))}</div><h3>${escapeHtml(article.title)}</h3><p>${escapeHtml(article.description)}</p></a>`;
  return html.slice(0, rowEnd) + card + html.slice(rowEnd);
}

function updateSitemap(slug, date) {
  const sitemap = fs.readFileSync(SITEMAP_PATH, 'utf8');
  const entry = `  <url>\n    <loc>https://www.nano-banana.live/blog/${slug}.html</loc>\n    <lastmod>${date}</lastmod>\n  </url>\n`;
  if (sitemap.includes(`/blog/${slug}.html`)) return sitemap.replace(new RegExp(`(<loc>https://www\\.nano-banana\\.live/blog/${slug}\\.html<\\/loc>\\s*<lastmod>).*?(<\\/lastmod>)`), `$1${date}$2`);
  return sitemap.replace('</urlset>', `${entry}</urlset>`);
}

async function main() {
  if (!ai) throw new Error('Missing NVIDIA_API_KEY.');
  const queue = readJson(QUEUE_PATH);
  const articles = readJson(ARTICLES_PATH);
  const seoResearch = readJson(SEO_RESEARCH_PATH);
  const existingSlugs = new Set(articles.map(article => article.slug));
  const item = queue.find(entry => entry.status === 'pending' && !existingSlugs.has(entry.slug));
  if (!item) {
    console.log('No pending blog keyword.');
    return;
  }
  const date = new Date().toISOString().slice(0, 10);
  const cluster = findSeoCluster(item, seoResearch);
  const article = await generateArticle(item, articles, cluster);
  const structuralScore = scoreArticle(article, item, cluster);
  if (structuralScore < 75) throw new Error(`Structural score ${structuralScore}/100 is below the 75-point publishing threshold.`);
  const htmlPath = path.join(BLOG_DIR, `${item.slug}.html`);
  if (fs.existsSync(htmlPath)) throw new Error(`Article already exists: ${item.slug}.html`);
  const backups = {
    articles: fs.readFileSync(ARTICLES_PATH, 'utf8'),
    queue: fs.readFileSync(QUEUE_PATH, 'utf8'),
    blog: fs.readFileSync(BLOG_INDEX_PATH, 'utf8'),
    sitemap: fs.readFileSync(SITEMAP_PATH, 'utf8'),
  };
  try {
    fs.writeFileSync(htmlPath, buildArticleHtml(article, item, date), 'utf8');
    articles.push({ slug: item.slug, publishDate: date, title: article.title, description: article.description, category: item.category, emoji: item.emoji, keyword: item.keyword, structuralScore, sourceAudit: 'automated-pass', sourceUrls: item.sourceUrls });
    item.status = 'done';
    item.structuralScore = structuralScore;
    item.sourceAudit = 'automated-pass';
    writeJson(ARTICLES_PATH, articles);
    writeJson(QUEUE_PATH, queue);
    fs.writeFileSync(BLOG_INDEX_PATH, updateBlogIndex(article, item, date), 'utf8');
    fs.writeFileSync(SITEMAP_PATH, updateSitemap(item.slug, date), 'utf8');
    console.log(`Prepared source-audited article ${item.slug}.html with structural score ${structuralScore}/100.`);
  } catch (error) {
    fs.writeFileSync(ARTICLES_PATH, backups.articles, 'utf8');
    fs.writeFileSync(QUEUE_PATH, backups.queue, 'utf8');
    fs.writeFileSync(BLOG_INDEX_PATH, backups.blog, 'utf8');
    fs.writeFileSync(SITEMAP_PATH, backups.sitemap, 'utf8');
    if (fs.existsSync(htmlPath)) fs.unlinkSync(htmlPath);
    throw error;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  main().catch(error => {
    console.error(error.message);
    process.exit(1);
  });
}

export {
  MODELS,
  VERIFIER_MODELS,
  classifyModelError,
  fetchSourceNotes,
  isAuthoritativeExternalSource,
  modelsForItem,
  normalizeDescription,
  normalizeTitle,
  parseModelList,
  sanitizeHtml,
  validateArticle,
};
