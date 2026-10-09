const SITE_URL = 'https://www.nano-banana.live';
const INDEX_START = '<!-- BLOG-INDEX:START -->';
const INDEX_END = '<!-- BLOG-INDEX:END -->';

function validateDate(date) {
  const parsed = new Date(date);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== date) {
    throw new Error('Invalid content modification date: ' + date);
  }
}

function escapeXml(value) {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function updateSitemapDates(source, changes) {
  if (!source.includes('</urlset>')) throw new Error('Sitemap urlset not found.');
  let result = source;
  for (const [url, date] of Object.entries(changes)) {
    validateDate(date);
    const parsed = new URL(url);
    if (parsed.origin !== SITE_URL || parsed.search || parsed.hash) {
      throw new Error('Unexpected sitemap URL: ' + url);
    }
    const loc = '<loc>' + escapeXml(url) + '</loc>';
    let matches = 0;
    result = result.replace(/<url>\s*[\s\S]*?<\/url>/g, block => {
      if (!block.includes(loc)) return block;
      matches++;
      if (/<lastmod>[^<]*<\/lastmod>/.test(block)) {
        return block.replace(/<lastmod>[^<]*<\/lastmod>/, '<lastmod>' + date + '</lastmod>');
      }
      return block.replace(loc, loc + '\n    <lastmod>' + date + '</lastmod>');
    });
    if (matches > 1) throw new Error('Duplicate sitemap URL: ' + url);
    if (!matches) {
      const entry = '  <url>\n    ' + loc + '\n    <lastmod>' + date + '</lastmod>\n  </url>\n';
      result = result.replace('</urlset>', entry + '</urlset>');
    }
  }
  return result;
}

function buildArticleIndex(articles) {
  const slugs = new Set();
  const entries = [...articles].sort((a, b) => b.publishDate.localeCompare(a.publishDate) || a.slug.localeCompare(b.slug));
  return '\n## Published articles\n\n' + entries.map(article => {
    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(article.slug) || slugs.has(article.slug)) {
      throw new Error('Invalid or duplicate article slug: ' + article.slug);
    }
    slugs.add(article.slug);
    validateDate(article.publishDate);
    if (article.modifiedDate) validateDate(article.modifiedDate);
    const title = article.title.replace(/\s+/g, ' ').trim().replace(/[\\[\]]/g, '\\$&');
    const updated = article.modifiedDate ? '; updated ' + article.modifiedDate : '';
    return '- [' + title + '](' + SITE_URL + '/blog/' + article.slug + '.html) (published ' + article.publishDate + updated + ')';
  }).join('\n') + '\n\n';
}

function updateArticleSummary(source, articles) {
  const start = source.indexOf(INDEX_START);
  const end = source.indexOf(INDEX_END);
  if (start < 0 || end < start || source.indexOf(INDEX_START, start + 1) !== -1 || source.indexOf(INDEX_END, end + 1) !== -1) {
    throw new Error('Expected exactly one published-article index in the AI summary.');
  }
  const newline = source.includes('\r\n') ? '\r\n' : '\n';
  const index = buildArticleIndex(articles).replace(/\n/g, newline);
  return source.slice(0, start + INDEX_START.length) + index + source.slice(end);
}

module.exports = { SITE_URL, updateSitemapDates, buildArticleIndex, updateArticleSummary };
