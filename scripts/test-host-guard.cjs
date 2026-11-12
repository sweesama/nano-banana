const assert = require('node:assert/strict');
const vm = require('node:vm');
const { CANONICAL_ORIGIN, HOST_GUARD_SCRIPT, applyHostGuard, absolutize } = require('./host-guard.cjs');

const source = HOST_GUARD_SCRIPT.slice('<script>'.length, -'</script>'.length);

function navigate(hostname, extra = {}) {
  const location = {
    hostname,
    pathname: extra.pathname ?? '/guides/quickstart.html',
    search: extra.search ?? '?q=1',
    hash: extra.hash ?? '#setup',
    replace(url) { this.target = url; },
  };
  vm.runInNewContext(source, { location });
  return location.target || null;
}

for (const host of ['nano-banana.live', 'www.nano-banana.live', 'localhost', '127.0.0.1', 'nano-banana-git-main.vercel.app', 'FOO.VERCEL.APP']) {
  assert.equal(navigate(host), null, host);
}
assert.equal(
  navigate('nanobanana.ai', { pathname: '/ar/', search: '?lang=ar', hash: '#api' }),
  CANONICAL_ORIGIN + '/ar/?lang=ar#api',
);
assert.equal(navigate('vercel.app'), CANONICAL_ORIGIN + '/guides/quickstart.html?q=1#setup');
assert.equal(navigate('notvercel.app'), CANONICAL_ORIGIN + '/guides/quickstart.html?q=1#setup');
assert.equal(navigate('www.nano-banana.live.evil.com'), CANONICAL_ORIGIN + '/guides/quickstart.html?q=1#setup');
assert.equal(navigate('evil.vercel.app.attacker.com'), CANONICAL_ORIGIN + '/guides/quickstart.html?q=1#setup');

const sample = '<!doctype html>\n<html>\n<head>\n  <meta charset="utf-8">\n  <link rel="canonical" href="/about.html">\n  <meta property="og:url" content="about.html">\n</head>\n<body></body>\n</html>\n';
const guarded = applyHostGuard(sample, CANONICAL_ORIGIN + '/about.html');
assert.equal(applyHostGuard(guarded, CANONICAL_ORIGIN + '/about.html'), guarded);
assert.ok(guarded.includes('<head>\n  ' + HOST_GUARD_SCRIPT + '\n'));
assert.match(guarded, /<link rel="canonical" href="https:\/\/www\.nano-banana\.live\/about\.html">/);
assert.match(guarded, /<meta property="og:url" content="https:\/\/www\.nano-banana\.live\/about\.html">/);
assert.equal(absolutize('https://nano-banana.live/faq.html', CANONICAL_ORIGIN + '/'), CANONICAL_ORIGIN + '/faq.html');

const bare = '<html><head><title>Missing</title></head><body><h1>Missing</h1></body></html>';
const completed = applyHostGuard(bare, CANONICAL_ORIGIN + '/404.html');
assert.match(completed, /rel="canonical" href="https:\/\/www\.nano-banana\.live\/404\.html"/);
assert.match(completed, /property="og:url" content="https:\/\/www\.nano-banana\.live\/404\.html"/);
assert.equal(completed.indexOf(HOST_GUARD_SCRIPT), completed.indexOf('<script>'));

console.log('Host guard tests passed.');
