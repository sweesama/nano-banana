const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { SITE_URL } = require('./content-metadata.cjs');
const { IMAGE_URL, HUBS, applyPageSeo, buildPageMetadata, breadcrumbsFor, renderBreadcrumbs, updateTopicHub } = require('./page-seo.cjs');
const { syncPageSeo } = require('./sync-page-seo.cjs');
const web = path.resolve(__dirname, '..', 'web');
const articles = JSON.parse(fs.readFileSync(path.join(web, 'blog', 'articles.json'), 'utf8'));
const graphFor = html => JSON.parse(html.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/)[1])['@graph'];
const crumbs = breadcrumbsFor(SITE_URL + '/blog/example.html', 'An example');
assert.equal(crumbs.length, 3);
assert.match(renderBreadcrumbs(crumbs, SITE_URL + '/blog/example.html'), /href="\.\.\/index\.html"/);
assert.match(renderBreadcrumbs(crumbs, SITE_URL + '/blog/example.html'), /aria-current="page"/);
assert.deepEqual(breadcrumbsFor(SITE_URL + '/', 'Home'), []);
const unsafe = buildPageMetadata({ title: 'A "title"', description: '</script><script>bad()</script>', canonical: SITE_URL + '/blog/example.html', breadcrumbs: crumbs,
  schemas: [{ '@type': 'Article', headline: 'An example', description: '</script><script>bad()</script>', datePublished: '2026-01-20', author: { '@type': 'Organization', name: 'Nano Banana' } }] });
assert.ok(unsafe.includes('A &quot;title&quot;'));
assert.doesNotMatch(unsafe, /<\/script><script>bad/);
assert.match(unsafe, /\\u003c/);
const nodes = graphFor(unsafe);
assert.equal(nodes.find(node => node['@type'] === 'Organization').url, SITE_URL + '/about.html');
assert.equal(nodes.find(node => node['@type'] === 'Article').datePublished, '2026-01-20');
assert.ok(!nodes.find(node => node['@type'] === 'Article').image);
const quotedPage = '<html><head><title>A guide</title><meta name="description" content="Google\'s hosted model &amp; local choices"><link rel="canonical" href="' + SITE_URL + '/guides/example.html"></head><body><nav></nav><h1>A guide &#8212; choices</h1></body></html>';
const quotedSeo = applyPageSeo(quotedPage, 'guides/example.html', []);
assert.equal(graphFor(quotedSeo).find(node => node['@type'] === 'TechArticle').description, "Google's hosted model & local choices");
assert.equal(graphFor(quotedSeo).find(node => node['@type'] === 'TechArticle').headline, 'A guide \u2014 choices');
for (const hub of HUBS) {
  const source = fs.readFileSync(path.join(web, hub.file), 'utf8').replace(/\r\n/g, '\n');
  assert.equal(updateTopicHub(source, articles, hub), source, hub.file);
  const added = [...articles, { slug: 'newly-published', title: 'A new Gemini image API guide', publishDate: '2026-10-10', category: 'API Tutorial' }];
  if (hub.groups.includes('api')) assert.match(updateTopicHub(source, added, hub), /newly-published\.html/);
}
function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).filter(entry => !['node_modules', '.diagnostics'].includes(entry.name))
    .flatMap(entry => entry.isDirectory() ? walk(path.join(dir, entry.name)) : [path.join(dir, entry.name)]);
}
const indexed = walk(web).filter(file => file.endsWith('.html') && !/name=["']robots["'][^>]+content=["'][^"']*noindex/i.test(fs.readFileSync(file, 'utf8')));
const incoming = new Map(indexed.map(file => [file, new Set()]));
for (const file of indexed) {
  const source = fs.readFileSync(file, 'utf8').replace(/\r\n/g, '\n');
  const relative = path.relative(web, file).replaceAll('\\', '/');
  assert.equal(applyPageSeo(source, relative, articles), source, 'Idempotent SEO sync: ' + relative);
  const canonical = source.match(/<link\s+rel=["']canonical["'][^>]+href=["']([^"']+)["']/i)[1];
  assert.ok(source.includes('property="og:url" content="' + canonical + '"'));
  assert.ok(source.includes('property="og:image" content="' + IMAGE_URL + '"'));
  const breadcrumb = graphFor(source).find(node => node['@type'] === 'BreadcrumbList');
  if (canonical !== SITE_URL + '/') {
    assert.equal(breadcrumb.itemListElement.at(-1).item, canonical);
    assert.match(source, /aria-label="Breadcrumb"/);
  }
  for (const match of source.matchAll(/<a\b[^>]*href=["']([^"']+)["']/gi)) {
    let url;
    try { url = new URL(match[1], canonical); } catch { continue; }
    if (url.origin !== SITE_URL) continue;
    let pathname = decodeURIComponent(url.pathname);
    if (pathname.endsWith('/')) pathname += 'index.html';
    const target = path.resolve(web, '.' + pathname);
    if (target !== file && incoming.has(target)) incoming.get(target).add(file);
  }
}
for (const article of articles) assert.ok(incoming.get(path.join(web, 'blog', article.slug + '.html'))?.size >= 2, 'Needs more than one incoming page: ' + article.slug);
assert.ok(incoming.get(path.join(web, 'benchmarks', 'portrait-editing.html')).size >= 2);
const png = fs.readFileSync(path.join(web, 'og.png'));
assert.equal(png.readUInt32BE(16), 1200);
assert.equal(png.readUInt32BE(20), 630);
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'nano-banana-page-seo-'));
try {
  fs.mkdirSync(path.join(dir, 'blog'));
  fs.mkdirSync(path.join(dir, 'node_modules'));
  fs.writeFileSync(path.join(dir, 'node_modules', 'dependency.html'), 'not a public site page');
  const home = '<html><head><title>Home</title><meta name="description" content="Description"><link rel="canonical" href="' + SITE_URL + '/"></head><body><nav></nav><h1>Home</h1><!-- TOPIC-LINKS:START --><!-- TOPIC-LINKS:END --></body></html>';
  fs.writeFileSync(path.join(dir, 'index.html'), home);
  fs.writeFileSync(path.join(dir, '404.html'), '<meta name="robots" content="noindex">');
  fs.writeFileSync(path.join(dir, 'blog', 'articles.json'), '[]');
  for (const name of ['llms.txt', 'llms-full.txt']) fs.writeFileSync(path.join(dir, name), '<!-- BLOG-INDEX:START --><!-- BLOG-INDEX:END -->');
  fs.writeFileSync(path.join(dir, 'sitemap.xml'), '<urlset><url><loc>' + SITE_URL + '/</loc><lastmod>2026-08-13</lastmod></url></urlset>');
  const originals = new Map(walk(dir).map(file => [file, fs.readFileSync(file, 'utf8')]));
  let writes = 0;
  assert.throws(() => syncPageSeo({ webDir: dir, date: '2026-10-09', write: (...args) => {
    if (++writes === 2) throw new Error('Simulated SEO write failure');
    fs.writeFileSync(...args);
  } }), /Simulated/);
  for (const [file, original] of originals) assert.equal(fs.readFileSync(file, 'utf8'), original);
  syncPageSeo({ webDir: dir, date: '2026-10-09' });
  assert.match(fs.readFileSync(path.join(dir, 'sitemap.xml'), 'utf8'), /2026-10-09/);
  assert.deepEqual(syncPageSeo({ webDir: dir }), []);
  assert.equal(fs.readFileSync(path.join(dir, '404.html'), 'utf8'), '<meta name="robots" content="noindex">');
  assert.equal(fs.readFileSync(path.join(dir, 'node_modules', 'dependency.html'), 'utf8'), 'not a public site page');
} finally {
  const target = fs.realpathSync(dir);
  assert.equal(path.dirname(target), fs.realpathSync(os.tmpdir()));
  assert.ok(path.basename(target).startsWith('nano-banana-page-seo-'));
  fs.rmSync(target, { recursive: true, force: true });
}
console.log('Page SEO tests passed: ' + indexed.length + ' indexable pages, ' + articles.length + ' articles with multiple incoming pages, and atomic sync.');
