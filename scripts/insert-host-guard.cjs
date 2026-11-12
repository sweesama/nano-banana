const fs = require('node:fs');
const path = require('node:path');
const { applyHostGuard, pageUrl } = require('./host-guard.cjs');

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true })
    .filter((entry) => !['node_modules', '.diagnostics'].includes(entry.name))
    .flatMap((entry) => {
      const target = path.join(dir, entry.name);
      return entry.isDirectory() ? walk(target) : [target];
    });
}

function insertHostGuards(webRoot) {
  const changed = [];
  for (const file of walk(webRoot).filter((item) => item.endsWith('.html'))) {
    const original = fs.readFileSync(file, 'utf8');
    const updated = applyHostGuard(original, pageUrl(file, webRoot));
    if (updated !== original) {
      fs.writeFileSync(file, updated);
      changed.push(path.relative(webRoot, file).replaceAll('\\', '/'));
    }
  }
  return changed;
}

if (require.main === module) {
  const webRoot = path.resolve(__dirname, '..', 'web');
  const changed = insertHostGuards(webRoot);
  console.log('Host guard updated ' + changed.length + ' HTML file(s).');
  for (const file of changed) console.log('- ' + file);
}

module.exports = { insertHostGuards };
