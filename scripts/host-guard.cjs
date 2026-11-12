const path = require('node:path');

const CANONICAL_ORIGIN = 'https://www.nano-banana.live';

// vercel.json sends Content-Security-Policy-Report-Only with
// script-src 'self' 'unsafe-inline', so this inline script is allowed.
// Keep the snippet inline: a reverse proxy that copies the HTML still
// executes it, and it runs before AdSense and Analytics.
const HOST_GUARD_SCRIPT = '<script>(function(){var h=location.hostname.toLowerCase();if(h==="nano-banana.live"||h==="www.nano-banana.live"||h==="localhost"||h==="127.0.0.1"||h.endsWith(".vercel.app"))return;location.replace("https://www.nano-banana.live"+location.pathname+location.search+location.hash)})();</script>';

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function pageUrl(file, webRoot) {
  const pathname = path.relative(webRoot, file).replaceAll('\\', '/');
  if (pathname === 'index.html') return CANONICAL_ORIGIN + '/';
  if (pathname.endsWith('/index.html')) return CANONICAL_ORIGIN + '/' + pathname.slice(0, -'index.html'.length);
  return CANONICAL_ORIGIN + '/' + pathname;
}

function absolutize(href, baseUrl) {
  let url;
  try {
    url = new URL(href, baseUrl);
  } catch {
    return href;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return href;
  const host = url.hostname.toLowerCase();
  if (host !== 'nano-banana.live' && host !== 'www.nano-banana.live') return href;
  return CANONICAL_ORIGIN + url.pathname + url.search + url.hash;
}

function tagAttr(tag, attr) {
  return tag.match(new RegExp('\\b' + attr + '=(["\'])([^"\']*)\\1', 'i'))?.[2] || '';
}

function replaceAttr(tag, attr, value) {
  return tag.replace(new RegExp('\\b' + attr + '=(["\'])([^"\']*)\\1', 'i'), attr + '=$1' + value + '$1');
}

function insertHostGuard(html) {
  const newline = html.includes('\r\n') ? '\r\n' : '\n';
  const stripped = html.replace(new RegExp('(?:\\r?\\n)[ \\t]*' + escapeRegExp(HOST_GUARD_SCRIPT)), '');
  if (!/<head\b/i.test(stripped)) throw new Error('HTML page is missing <head>');
  return stripped.replace(/<head\b[^>]*>/i, (open) => open + newline + '  ' + HOST_GUARD_SCRIPT);
}

function ensureAbsoluteUrls(html, baseUrl) {
  let next = html.replace(/<link\b[^>]*\brel=["']canonical["'][^>]*>/gi, (tag) => {
    const href = tagAttr(tag, 'href');
    if (!href) return tag;
    const absolute = absolutize(href, baseUrl);
    return absolute === href ? tag : replaceAttr(tag, 'href', absolute);
  });
  next = next.replace(/<meta\b[^>]*\bproperty=["']og:url["'][^>]*>/gi, (tag) => {
    const content = tagAttr(tag, 'content');
    if (!content) return tag;
    const absolute = absolutize(content, baseUrl);
    return absolute === content ? tag : replaceAttr(tag, 'content', absolute);
  });

  const canonicalTag = next.match(/<link\b[^>]*\brel=["']canonical["'][^>]*>/i);
  const canonical = canonicalTag ? absolutize(tagAttr(canonicalTag[0], 'href'), baseUrl) : baseUrl;
  const newline = next.includes('\r\n') ? '\r\n' : '\n';
  if (!canonicalTag) {
    next = next.replace(/<\/head>/i, '  <link rel="canonical" href="' + canonical + '">' + newline + '</head>');
  }
  if (!/<meta\b[^>]*\bproperty=["']og:url["']/i.test(next)) {
    const og = '<meta property="og:url" content="' + canonical + '">';
    const link = next.match(/<link\b[^>]*\brel=["']canonical["'][^>]*>/i);
    next = link
      ? next.replace(link[0], link[0] + newline + '  ' + og)
      : next.replace(/<\/head>/i, '  ' + og + newline + '</head>');
  }
  return next;
}

function applyHostGuard(html, baseUrl) {
  return ensureAbsoluteUrls(insertHostGuard(html), baseUrl);
}

module.exports = {
  CANONICAL_ORIGIN,
  HOST_GUARD_SCRIPT,
  absolutize,
  applyHostGuard,
  insertHostGuard,
  pageUrl,
};
