const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { SITE_URL, updateSitemapDates, updateArticleSummary } = require('./content-metadata.cjs');
const { refreshBenchmarkHtml, writeBenchmarkSnapshot } = require('./sync-benchmarks.js');

const sitemap = '<urlset><url><loc>' + SITE_URL + '/blog.html</loc><lastmod>2026-08-13</lastmod></url><url><loc>' + SITE_URL + '/benchmarks/</loc><lastmod>2026-04-25</lastmod></url></urlset>';
const changes = { [SITE_URL + '/blog/example.html']: '2026-10-09', [SITE_URL + '/blog.html']: '2026-10-09' };
const updated = updateSitemapDates(sitemap, changes);
assert.match(updated, /blog\.html<\/loc><lastmod>2026-10-09/);
assert.match(updated, /benchmarks\/<\/loc><lastmod>2026-04-25/);
assert.equal((updated.match(/example\.html/g) || []).length, 1);
assert.equal(updateSitemapDates(updated, changes), updated);
assert.throws(() => updateSitemapDates(sitemap, { [SITE_URL + '/']: '2026-02-30' }), /Invalid/);
assert.throws(() => updateSitemapDates(sitemap, { 'https://example.com/': '2026-10-09' }), /Unexpected/);
assert.throws(() => updateSitemapDates('<urlset>', changes), /urlset/);
assert.throws(() => updateSitemapDates(sitemap.replace('</urlset>', '<url><loc>' + SITE_URL + '/blog.html</loc></url></urlset>'), changes), /Duplicate/);
assert.match(updateSitemapDates('<urlset><url><loc>' + SITE_URL + '/blog.html</loc></url></urlset>', changes), /<lastmod>2026-10-09/);

const summary = '# Identity\n\n<!-- BLOG-INDEX:START -->\nold content\n<!-- BLOG-INDEX:END -->\n\nPolicy unchanged.';
const articles = [
  { slug: 'older', title: 'Older [model] guide', publishDate: '2026-09-01', modifiedDate: '2026-10-09' },
  { slug: 'newer', title: 'Newer guide', publishDate: '2026-10-08' },
];
const indexed = updateArticleSummary(summary, articles);
assert.ok(indexed.indexOf('/newer.html') < indexed.indexOf('/older.html'));
assert.ok(indexed.startsWith('# Identity\n\n'));
assert.ok(indexed.endsWith('Policy unchanged.'));
assert.ok(indexed.includes('Older \\[model\\] guide'));
assert.match(indexed, /published 2026-09-01; updated 2026-10-09/);
assert.equal(updateArticleSummary(indexed, articles), indexed);
const windowsSummary = indexed.replace(/\n/g, '\r\n');
assert.equal(updateArticleSummary(windowsSummary, articles), windowsSummary);
assert.equal(articles[0].slug, 'older');
assert.throws(() => updateArticleSummary('no markers', articles), /exactly one/);
assert.throws(() => updateArticleSummary(summary + '<!-- BLOG-INDEX:END -->', articles), /exactly one/);
assert.throws(() => updateArticleSummary(summary, [...articles, articles[0]]), /duplicate/);
assert.throws(() => updateArticleSummary(summary, [{ ...articles[0], slug: '../private' }]), /Invalid/);

const fixture = '<section id="text-to-image"><table><tbody>old text</tbody></table></section>' +
  '<section id="image-editing"><table><tbody>old editing</tbody></table></section>' +
  'Data refreshed: <strong>April 1, 2026</strong>';
const model = { name: 'A & B <model>', elo: 1234, model_creator: { name: 'Unknown' } };
const html = refreshBenchmarkHtml(fixture, [model], [model], new Date('2026-10-09T00:00:00Z'));
assert.match(html, /Data refreshed: <strong>October 9, 2026<\/strong>/);
assert.match(html, /A &amp; B &lt;model&gt;/);
assert.doesNotMatch(html, /old text|old editing/);

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'nano-banana-metadata-'));
try {
  fs.mkdirSync(path.join(dir, 'benchmarks'));
  const htmlPath = path.join(dir, 'benchmarks', 'index.html');
  const sitemapPath = path.join(dir, 'sitemap.xml');
  fs.writeFileSync(htmlPath, fixture);
  fs.writeFileSync(sitemapPath, sitemap);
  writeBenchmarkSnapshot(html, '2026-10-09', dir);
  assert.equal(fs.readFileSync(htmlPath, 'utf8'), html);
  assert.match(fs.readFileSync(sitemapPath, 'utf8'), /benchmarks\/<\/loc><lastmod>2026-10-09/);
  const savedSitemap = fs.readFileSync(sitemapPath, 'utf8');
  let writes = 0;
  assert.throws(() => writeBenchmarkSnapshot('partial write', '2026-10-10', dir, (...args) => {
    if (++writes === 2) throw new Error('Simulated sitemap write failure');
    fs.writeFileSync(...args);
  }), /Simulated/);
  assert.equal(fs.readFileSync(htmlPath, 'utf8'), html);
  assert.equal(fs.readFileSync(sitemapPath, 'utf8'), savedSitemap);
} finally {
  const target = fs.realpathSync(dir);
  assert.equal(path.dirname(target), fs.realpathSync(os.tmpdir()));
  assert.ok(path.basename(target).startsWith('nano-banana-metadata-'));
  fs.rmSync(target, { recursive: true, force: true });
}

const web = path.resolve(__dirname, '..', 'web');
for (const [file, size] of [['favicon.png', 96], ['favicon-32x32.png', 32], ['favicon-16x16.png', 16], ['apple-touch-icon.png', 180]]) {
  const buffer = fs.readFileSync(path.join(web, file));
  assert.equal(buffer.readUInt32BE(16), size, file);
  assert.equal(buffer.readUInt32BE(20), size, file);
  assert.ok(buffer.length < 100000, file + ' must be a small icon');
}
const ico = fs.readFileSync(path.join(web, 'favicon.ico'));
assert.equal(ico.readUInt16LE(2), 1);
assert.equal(ico.readUInt16LE(4), 1);
assert.ok(ico.length < 10000);
for (const file of ['nano-640.webp', 'nano-960.webp', 'nano-1256.webp', 'nano-960.jpg']) {
  assert.ok(fs.statSync(path.join(web, file)).size < 250000, file + ' must stay lightweight');
}
console.log('Content metadata, benchmark rollback, and image asset tests passed.');
