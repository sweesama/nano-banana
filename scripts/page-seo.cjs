const path = require('node:path');
const { SITE_URL } = require('./content-metadata.cjs');

const IMAGE_URL = SITE_URL + '/og.png';
const ORGANIZATION_ID = SITE_URL + '/#publisher';
const HUBS = [
  { file: 'index.html', groups: ['api', 'local', 'cloud'], limit: 3 },
  { file: 'guides/quickstart.html', groups: ['api', 'local', 'cloud', 'benchmarks', 'prompting'], collapsed: true },
  { file: 'guides/cloud-deployment.html', groups: ['cloud', 'local'], limit: 5 },
  { file: 'benchmarks/index.html', groups: ['benchmarks'] },
  { file: 'faq.html', groups: ['api', 'prompting'], limit: 6 },
];
const TOPICS = {
  api: { title: 'Cloud API and troubleshooting', categories: ['API Tutorial', 'API Guide', 'Integration', 'Troubleshooting', 'Workflows', 'Support'] },
  local: { title: 'Offline models and hardware', categories: ['Local Models', 'Hardware'] },
  cloud: { title: 'Cloud GPUs and deployment choices', categories: ['Cloud GPU', 'Cloud & Budget', 'Comparisons'] },
  benchmarks: { title: 'Benchmarks and evidence', categories: ['Benchmarks'] },
  prompting: { title: 'Editing prompts and recipes', categories: ['Prompting', 'Recipes'] },
};

function escapeHtml(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function decodeHtml(value) {
  const entities = { amp: '&', quot: '"', '#39': "'", lt: '<', gt: '>' };
  return value.replace(/&(amp|quot|#39|lt|gt);/g, (_, name) => entities[name])
    .replace(/&#(x[0-9a-f]+|\d+);/gi, (entity, number) => {
      const point = number.toLowerCase().startsWith('x') ? parseInt(number.slice(1), 16) : Number(number);
      return point > 0 && point <= 0x10ffff ? String.fromCodePoint(point) : entity;
    });
}

function relativeHref(canonical, target) {
  const current = new URL(canonical).pathname;
  const destination = new URL(target).pathname;
  const from = path.posix.dirname(current.endsWith('/') ? current + 'index.html' : current);
  return path.posix.relative(from, destination.endsWith('/') ? destination + 'index.html' : destination) || 'index.html';
}

function breadcrumbsFor(canonical, label) {
  const pathname = new URL(canonical).pathname;
  if (pathname === '/') return [];
  const items = [{ name: 'Home', url: SITE_URL + '/' }];
  if (pathname.startsWith('/blog/')) items.push({ name: 'Blog', url: SITE_URL + '/blog.html' });
  if (pathname.startsWith('/guides/') && pathname !== '/guides/quickstart.html') items.push({ name: 'Quickstart', url: SITE_URL + '/guides/quickstart.html' });
  if (pathname.startsWith('/benchmarks/') && pathname !== '/benchmarks/') items.push({ name: 'Benchmarks', url: SITE_URL + '/benchmarks/' });
  items.push({ name: label, url: canonical });
  return items;
}

function renderBreadcrumbs(items, canonical) {
  if (!items.length) return '';
  return '<!-- BREADCRUMBS:START -->\n<nav class="breadcrumbs" aria-label="Breadcrumb"><ol>' +
    items.map((item, index) => '<li>' + (index === items.length - 1
      ? '<span aria-current="page">' + escapeHtml(item.name) + '</span>'
      : '<a href="' + escapeHtml(relativeHref(canonical, item.url)) + '">' + escapeHtml(item.name) + '</a>') + '</li>').join('') +
    '</ol></nav>\n<!-- BREADCRUMBS:END -->';
}

function buildPageMetadata({ title, description, canonical, schemas = [], breadcrumbs = [] }) {
  const organization = {
    '@type': 'Organization', '@id': ORGANIZATION_ID, name: 'Nano Banana', url: SITE_URL + '/about.html',
    description: 'Independent API and local-model reference, not an official Google website.',
    logo: { '@type': 'ImageObject', url: SITE_URL + '/apple-touch-icon.png', width: 180, height: 180 },
  };
  const nodes = schemas.filter(node => node['@type'] !== 'BreadcrumbList' && node['@id'] !== ORGANIZATION_ID).map(node => {
    const copy = { ...node };
    delete copy['@context'];
    if (copy['@type'] === 'WebSite') {
      copy['@id'] = SITE_URL + '/#website';
      copy.description = description;
      copy.publisher = { '@id': ORGANIZATION_ID };
    }
    if (['Article', 'TechArticle'].includes(copy['@type'])) {
      copy.publisher = { '@id': ORGANIZATION_ID };
      if (!copy.author || copy.author.name === 'Nano Banana' || copy.author['@id'] === ORGANIZATION_ID) {
        copy.author = { '@id': ORGANIZATION_ID };
      }
      copy.mainEntityOfPage = { '@type': 'WebPage', '@id': canonical };
    }
    return copy;
  });
  if (breadcrumbs.length) nodes.push({
    '@type': 'BreadcrumbList',
    itemListElement: breadcrumbs.map((item, index) => ({ '@type': 'ListItem', position: index + 1, name: item.name, item: item.url })),
  });
  const graph = JSON.stringify({ '@context': 'https://schema.org', '@graph': [organization, ...nodes] }).replace(/</g, '\\u003c');
  const article = nodes.some(node => ['Article', 'TechArticle'].includes(node['@type']));
  const tags = {
    'og:title': title, 'og:description': description, 'og:type': article ? 'article' : 'website',
    'og:url': canonical, 'og:image': IMAGE_URL, 'og:image:width': '1200', 'og:image:height': '630',
    'og:image:alt': 'Nano Banana: image editing, benchmarks and quickstart', 'og:site_name': 'Nano Banana',
  };
  return '<!-- PAGE-METADATA:START -->\n' +
    Object.entries(tags).map(([name, value]) => '<meta property="' + name + '" content="' + escapeHtml(value) + '">').join('\n') +
    '\n<meta name="twitter:card" content="summary_large_image">\n<meta name="twitter:title" content="' + escapeHtml(title) + '">\n' +
    '<meta name="twitter:description" content="' + escapeHtml(description) + '">\n<meta name="twitter:image" content="' + IMAGE_URL + '">\n' +
    '<script type="application/ld+json">' + graph + '</script>\n<!-- PAGE-METADATA:END -->';
}

function applyPageSeo(source, file, articles, { modifiedDate } = {}) {
  const title = decodeHtml(source.match(/<title>([^<]+)<\/title>/i)?.[1] || '');
  const description = decodeHtml(source.match(/<meta\s+name=["']description["'][^>]+content=(["'])([\s\S]*?)\1/i)?.[2] || '');
  const canonical = source.match(/<link\s+rel=["']canonical["'][^>]+href=["']([^"']+)["']/i)?.[1];
  const h1 = decodeHtml(source.match(/<h1\b[^>]*>([\s\S]*?)<\/h1>/i)?.[1].replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim() || '');
  if (!title || !description || !canonical || !h1) throw new Error('Missing page metadata: ' + file);
  const article = articles.find(item => file === 'blog/' + item.slug + '.html');
  let schemas = [...source.matchAll(/<script\b[^>]*type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi)].flatMap(match => {
    const node = JSON.parse(match[1]);
    return node['@graph'] || [node];
  }).filter(node => node['@type'] !== 'BreadcrumbList' && node['@id'] !== ORGANIZATION_ID);
  if (!schemas.length) {
    const type = article ? 'Article' : file === 'index.html' ? 'WebSite' : file.startsWith('guides/') ? 'TechArticle' : file === 'benchmarks/index.html' ? 'CollectionPage' : 'WebPage';
    schemas = [{ '@type': type, name: h1, description, url: canonical }];
  }
  schemas = schemas.map(node => {
    if (!['Article', 'TechArticle'].includes(node['@type'])) return node;
    const updated = { ...node, headline: h1, description };
    if (article) {
      updated.datePublished ||= article.publishDate;
      updated.dateModified = article.modifiedDate || updated.dateModified || article.reviewedAt || article.publishDate;
    }
    if (modifiedDate) updated.dateModified = modifiedDate;
    return updated;
  });
  const label = file === 'guides/quickstart.html' ? 'Quickstart' : file === 'blog.html' ? 'Blog' : file === 'benchmarks/index.html' ? 'Benchmarks' : h1;
  const breadcrumbs = breadcrumbsFor(canonical, label);
  const metadata = buildPageMetadata({ title, description, canonical, schemas, breadcrumbs });
  const placeholder = '<!-- PAGE-METADATA:INSERT -->';
  const hasManagedMetadata = /<!-- PAGE-METADATA:START -->/.test(source);
  let result = source.replace(/<!-- PAGE-METADATA:START -->[\s\S]*?<!-- PAGE-METADATA:END -->/g, placeholder)
    .replace(/[ \t]*<meta\b[^>]*(?:property|name)=["'](?:og:|twitter:)[^"']+["'][^>]*>[ \t]*\r?\n?/gmi, '')
    .replace(/[ \t]*<script\b[^>]*type=["']application\/ld\+json["'][^>]*>[\s\S]*?<\/script>[ \t]*\r?\n?/gmi, '');
  result = hasManagedMetadata ? result.replace(placeholder, metadata) : result.replace(/<\/head>/i, metadata + '\n</head>');
  const visible = renderBreadcrumbs(breadcrumbs, canonical);
  if (/<!-- BREADCRUMBS:START -->/.test(result)) {
    result = result.replace(/<!-- BREADCRUMBS:START -->[\s\S]*?<!-- BREADCRUMBS:END -->/, visible);
  } else if (visible) {
    result = result.replace('</nav>', '</nav>\n' + visible);
  }
  return result;
}

function updateTopicHub(source, articles, hub) {
  const canonical = source.match(/<link\s+rel=["']canonical["'][^>]+href=["']([^"']+)["']/i)?.[1];
  const start = '<!-- TOPIC-LINKS:START -->';
  const end = '<!-- TOPIC-LINKS:END -->';
  const startAt = source.indexOf(start);
  const endAt = source.indexOf(end);
  if (!canonical || startAt < 0 || endAt < startAt || source.indexOf(start, startAt + 1) !== -1 || source.indexOf(end, endAt + 1) !== -1) {
    throw new Error('Expected exactly one topic hub marker pair: ' + hub.file);
  }
  const sorted = [...articles].sort((a, b) => b.publishDate.localeCompare(a.publishDate) || a.slug.localeCompare(b.slug));
  const groups = hub.groups.map(key => {
    const topic = TOPICS[key];
    const selected = sorted.filter(article => topic.categories.includes(article.category)).slice(0, hub.limit || sorted.length);
    const links = selected.map(article => '<li><a href="' + escapeHtml(relativeHref(canonical, SITE_URL + '/blog/' + article.slug + '.html')) + '">' + escapeHtml(article.title) + '</a></li>');
    if (key === 'local') {
      links.push('<li><a href="' + relativeHref(canonical, SITE_URL + '/guides/hidream-o1-image.html') + '">HiDream-O1-Image installation guide</a></li>');
      links.push('<li><a href="' + relativeHref(canonical, SITE_URL + '/guides/z-image.html') + '">Z-Image model guide</a></li>');
    }
    if (key === 'benchmarks') links.push('<li><a href="' + relativeHref(canonical, SITE_URL + '/benchmarks/portrait-editing.html') + '">Inspect a disclosed image-editing example</a></li>');
    return '<div><h3>' + topic.title + '</h3><ul>' + links.join('') + '</ul></div>';
  }).join('\n');
  const open = hub.collapsed
    ? '<details class="section guide-index" id="related-guides"><summary><h2>Guides by task</h2></summary>'
    : '<section class="section guide-index" id="related-guides"><h2>Guides by task</h2>';
  const close = hub.collapsed ? '</details>' : '</section>';
  const section = start + '\n' + open + '<div class="guide-index-grid">\n' + groups + '\n</div>' + close + '\n' + end;
  return source.replace(/<!-- TOPIC-LINKS:START -->[\s\S]*?<!-- TOPIC-LINKS:END -->/, section);
}

module.exports = { IMAGE_URL, HUBS, applyPageSeo, buildPageMetadata, breadcrumbsFor, renderBreadcrumbs, updateTopicHub };
