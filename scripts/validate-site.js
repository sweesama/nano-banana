const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const WEB = path.join(ROOT, 'web');
const errors = [];
const SITE_HOST = 'www.nano-banana.live';
const NOINDEX_ALLOWLIST = new Set(['web/404.html', 'web/blog/run-ai-image-models-without-gpu.html']);
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

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const target = path.join(dir, entry.name);
    return entry.isDirectory() ? walk(target) : [target];
  });
}

function rel(file) {
  return path.relative(ROOT, file).replaceAll('\\', '/');
}

function count(source, pattern) {
  return (source.match(pattern) || []).length;
}

function add(file, message) {
  errors.push(`${rel(file)}: ${message}`);
}

function sourceHost(url) {
  try {
    return new URL(url).hostname.toLowerCase();
  } catch {
    return '';
  }
}

function hasAuthoritativeExternalSource(urls) {
  return urls.some((url) => {
    const host = sourceHost(url);
    return host !== SITE_HOST && AUTHORITATIVE_SOURCE_HOSTS.has(host);
  });
}

function checkBalanced(file, html, tag) {
  const opens = count(html, new RegExp(`<${tag}\\b`, 'gi'));
  const closes = count(html, new RegExp(`</${tag}>`, 'gi'));
  if (opens !== closes) add(file, `unbalanced <${tag}> tags (${opens} open, ${closes} close)`);
}

function expectedCanonical(file) {
  const pathname = path.relative(WEB, file).replaceAll('\\', '/');
  if (pathname === 'index.html') return 'https://www.nano-banana.live/';
  if (pathname.endsWith('/index.html')) return `https://www.nano-banana.live/${pathname.slice(0, -'index.html'.length)}`;
  return `https://www.nano-banana.live/${pathname}`;
}

function resolveInternalHref(file, href) {
  const clean = href.split('#')[0].split('?')[0];
  if (!clean || /^(?:https?:|mailto:|tel:|javascript:)/i.test(clean)) return null;
  const base = clean.startsWith('/') ? WEB : path.dirname(file);
  let target = path.resolve(base, clean.replace(/^\//, ''));
  if (clean === '/') target = path.join(WEB, 'index.html');
  if (clean.endsWith('/')) target = path.join(target, 'index.html');
  return target;
}

const htmlFiles = walk(WEB).filter((file) => file.endsWith('.html'));
for (const file of htmlFiles) {
  const html = fs.readFileSync(file, 'utf8');
  const is404 = path.basename(file) === '404.html';

  if (count(html, /<title\b/gi) !== 1) add(file, 'must contain exactly one <title>');
  if (count(html, /<h1\b/gi) !== 1) add(file, 'must contain exactly one <h1>');
  if (!is404 && !/<meta\s+name=["']description["'][^>]+content=["'][^"']+/i.test(html)) add(file, 'missing non-empty meta description');
  if (!is404 && !/<link\s+rel=["']canonical["'][^>]+href=["']https:\/\/www\.nano-banana\.live\//i.test(html)) add(file, 'missing production canonical URL');
  if (!is404) {
    const canonical = html.match(/<link\s+rel=["']canonical["'][^>]+href=["']([^"']+)["']/i)?.[1];
    if (canonical && canonical !== expectedCanonical(file) && !NOINDEX_ALLOWLIST.has(rel(file))) add(file, `canonical mismatch: expected ${expectedCanonical(file)}, found ${canonical}`);
  }
  if (/name=["']robots["'][^>]+content=["'][^"']*noindex/i.test(html) && !NOINDEX_ALLOWLIST.has(rel(file))) add(file, 'unexpected noindex on a published page');

  for (const tag of ['a', 'pre', 'code']) checkBalanced(file, html, tag);
  if (/<a\b[^>]*>(?:(?!<\/a>)[\s\S])*<a\b/i.test(html)) add(file, 'nested anchor detected');
  if (/(?:href|src)=["'][^"']*[\r\n][^"']*["']/i.test(html)) add(file, 'newline inside href/src attribute');
  for (const match of html.matchAll(/<pre\b[^>]*>\s*<code\b[^>]*>([\s\S]*?)<\/code>\s*<\/pre>/gi)) {
    if (/<(?:link|script|style|img|div|section|article|a)\b/i.test(match[1])) add(file, 'structural HTML appears unescaped inside a code block');
  }

  for (const match of html.matchAll(/<script\s+type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi)) {
    try {
      JSON.parse(match[1]);
    } catch (error) {
      add(file, `invalid JSON-LD (${error.message})`);
    }
  }

  for (const match of html.matchAll(/<a\b[^>]*\bhref=["']([^"']*)["'][^>]*>/gi)) {
    const href = match[1];
    if (href === '#' && !/(?:data-clip|data-fullsrc)=/i.test(match[0])) add(file, 'placeholder href="#" is not allowed');
    const target = resolveInternalHref(file, href);
    if (target && !fs.existsSync(target)) add(file, `broken internal link: ${href}`);
  }
}

const factualFiles = [
  'web/index.html',
  'web/faq.html',
  'web/prompts/index.html',
  'web/guides/quickstart.html',
  'web/llms.txt',
  'web/llms-full.txt',
  'web/blog/hardware-requirements.html',
  'web/blog/how-to-run-locally.html',
  'web/blog/top-prompt-recipes.html',
].map((file) => path.join(ROOT, file));
const bannedLabels = ['Qwen-Image 2.1', 'Hunyuan 3.1', 'Gemini 3.0'];
for (const file of factualFiles) {
  const content = fs.readFileSync(file, 'utf8');
  for (const label of bannedLabels) {
    if (content.includes(label)) add(file, `unsupported or stale model label: ${label}`);
  }
}

const sitemapPath = path.join(WEB, 'sitemap.xml');
const sitemap = fs.readFileSync(sitemapPath, 'utf8');
const sitemapUrls = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((match) => match[1]);
if (new Set(sitemapUrls).size !== sitemapUrls.length) add(sitemapPath, 'contains duplicate URLs');
for (const url of sitemapUrls) {
  const pathname = new URL(url).pathname;
  let target;
  if (pathname === '/') {
    target = path.join(WEB, 'index.html');
  } else {
    target = path.join(WEB, pathname.replace(/^\//, ''));
    if (pathname.endsWith('/')) target = path.join(target, 'index.html');
  }
  if (!fs.existsSync(target)) add(sitemapPath, `URL has no matching static file: ${url}`);
}

const articlesPath = path.join(WEB, 'blog', 'articles.json');
try {
  const articles = JSON.parse(fs.readFileSync(articlesPath, 'utf8'));
  const slugs = articles.map((article) => article.slug);
  if (new Set(slugs).size !== slugs.length) add(articlesPath, 'contains duplicate slugs');
  for (const slug of slugs) {
    if (!fs.existsSync(path.join(WEB, 'blog', `${slug}.html`))) add(articlesPath, `missing article page for ${slug}`);
  }
  for (const article of articles) {
    if (!article.sourceUrls) continue;
    if (!Array.isArray(article.sourceUrls) || article.sourceUrls.length < 2) add(articlesPath, `${article.slug} needs at least two source URLs`);
    if (!hasAuthoritativeExternalSource(article.sourceUrls || [])) add(articlesPath, `${article.slug} needs an approved authoritative external source`);
    const pagePath = path.join(WEB, 'blog', `${article.slug}.html`);
    if (fs.existsSync(pagePath)) {
      const page = fs.readFileSync(pagePath, 'utf8');
      for (const url of article.sourceUrls || []) {
        if (sourceHost(url) !== SITE_HOST && !page.includes(url)) add(pagePath, `missing declared source link: ${url}`);
      }
    }
  }
} catch (error) {
  add(articlesPath, `invalid JSON (${error.message})`);
}

const queuePath = path.join(WEB, 'blog', 'queue.json');
try {
  const queue = JSON.parse(fs.readFileSync(queuePath, 'utf8'));
  const slugs = queue.map((item) => item.slug);
  if (new Set(slugs).size !== slugs.length) add(queuePath, 'contains duplicate slugs');
  for (const item of queue.filter((entry) => entry.status === 'pending')) {
    if (!Array.isArray(item.sourceUrls) || item.sourceUrls.length < 2) add(queuePath, `${item.slug} needs at least two source URLs`);
    if (!hasAuthoritativeExternalSource(item.sourceUrls || [])) add(queuePath, `${item.slug} needs an approved authoritative external source before generation`);
  }
} catch (error) {
  add(queuePath, `invalid JSON (${error.message})`);
}

if (errors.length) {
  console.error(`Site validation failed with ${errors.length} issue(s):`);
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

console.log(`Site validation passed: ${htmlFiles.length} HTML files and ${sitemapUrls.length} sitemap URLs checked.`);
