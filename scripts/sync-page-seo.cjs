const fs = require('node:fs');
const path = require('node:path');
const { HUBS, applyPageSeo, updateTopicHub } = require('./page-seo.cjs');
const { updateArticleSummary, updateSitemapDates } = require('./content-metadata.cjs');

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).filter(entry => !['node_modules', '.diagnostics'].includes(entry.name))
    .flatMap(entry => entry.isDirectory() ? walk(path.join(dir, entry.name)) : [path.join(dir, entry.name)]);
}

function syncPageSeo({ webDir = path.resolve(__dirname, '..', 'web'), date, modifiedFiles = [], write = fs.writeFileSync } = {}) {
  const articles = JSON.parse(fs.readFileSync(path.join(webDir, 'blog', 'articles.json'), 'utf8'));
  const updates = new Map();
  const dates = {};
  for (const file of walk(webDir).filter(file => file.endsWith('.html'))) {
    const original = fs.readFileSync(file, 'utf8');
    if (/name=["']robots["'][^>]+content=["'][^"']*noindex/i.test(original)) continue;
    const relative = path.relative(webDir, file).replaceAll('\\', '/');
    const normalized = original.replace(/\r\n/g, '\n');
    const hub = HUBS.find(item => item.file === relative);
    const linked = hub ? updateTopicHub(normalized, articles, hub) : normalized;
    const modified = date && (modifiedFiles.includes(relative) || linked !== normalized) ? date : undefined;
    const updated = applyPageSeo(linked, relative, articles, { modifiedDate: modified });
    if (updated !== normalized) updates.set(file, updated);
    if (modified) {
      const canonical = updated.match(/<link\s+rel=["']canonical["'][^>]+href=["']([^"']+)["']/i)?.[1];
      dates[canonical] = date;
    }
  }
  for (const name of ['llms.txt', 'llms-full.txt']) {
    const file = path.join(webDir, name);
    const original = fs.readFileSync(file, 'utf8');
    const updated = updateArticleSummary(original, articles);
    if (updated !== original) updates.set(file, updated);
  }
  if (Object.keys(dates).length) {
    const file = path.join(webDir, 'sitemap.xml');
    updates.set(file, updateSitemapDates(fs.readFileSync(file, 'utf8'), dates));
  }
  const backups = new Map([...updates.keys()].map(file => [file, fs.readFileSync(file, 'utf8')]));
  try {
    for (const [file, source] of updates) write(file, source, 'utf8');
  } catch (error) {
    for (const [file, source] of backups) fs.writeFileSync(file, source, 'utf8');
    throw error;
  }
  return [...updates.keys()].map(file => path.relative(webDir, file).replaceAll('\\', '/'));
}

if (require.main === module) {
  const date = process.argv.includes('--date') ? process.argv[process.argv.indexOf('--date') + 1] : undefined;
  const modifiedFiles = process.argv.flatMap((value, index) => value === '--modified' ? [process.argv[index + 1]] : []);
  try {
    const changed = syncPageSeo({ date, modifiedFiles });
    console.log('SEO metadata synchronized: ' + changed.length + ' file(s).');
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}

module.exports = { syncPageSeo };
