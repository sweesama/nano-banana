const HOST = 'www.nano-banana.live';
const KEY = '8894b846bf144cbdb509ce481d85f7d9';
const KEY_LOCATION = `https://${HOST}/${KEY}.txt`;
const SITEMAP_URL = `https://${HOST}/sitemap.xml`;
const ENDPOINT = 'https://api.indexnow.org/indexnow';
const MAX_URLS = 10000;

function usage() {
  console.error('Usage: node scripts/indexnow.mjs <url...> | --sitemap');
  process.exit(1);
}

function decodeXml(value) {
  return value
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'");
}

function hostnameOf(url) {
  try {
    return new URL(url).hostname;
  } catch {
    return '';
  }
}

async function urlsFromSitemap() {
  const response = await fetch(SITEMAP_URL);
  if (!response.ok) {
    throw new Error(`Failed to read sitemap (${response.status} ${SITEMAP_URL})`);
  }
  const xml = await response.text();
  const urls = [];
  for (const match of xml.matchAll(/<loc>\s*([^<\s]+)\s*<\/loc>/gi)) {
    const url = decodeXml(match[1].trim());
    if (hostnameOf(url) === HOST) urls.push(url);
  }
  return urls;
}

async function main() {
  const args = process.argv.slice(2);
  if (args.length === 0 || args.includes('-h') || args.includes('--help')) usage();

  const useSitemap = args.includes('--sitemap');
  const positional = [];
  for (const arg of args) {
    if (arg === '--sitemap') continue;
    if (arg.startsWith('-')) throw new Error(`Unknown option: ${arg}`);
    positional.push(arg);
  }
  if (!useSitemap && positional.length === 0) usage();

  for (const url of positional) {
    if (hostnameOf(url) === '') throw new Error(`Invalid URL: ${url}`);
  }

  const urls = [];
  if (useSitemap) urls.push(...await urlsFromSitemap());
  urls.push(...positional);

  const urlList = [...new Set(urls)];
  if (urlList.length === 0) throw new Error('No URLs to submit');

  let status = 0;
  for (let offset = 0; offset < urlList.length; offset += MAX_URLS) {
    const batch = urlList.slice(offset, offset + MAX_URLS);
    const response = await fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
      body: JSON.stringify({
        host: HOST,
        key: KEY,
        keyLocation: KEY_LOCATION,
        urlList: batch,
      }),
    });
    status = response.status;
    const text = await response.text();
    console.log(`IndexNow ${status} (${batch.length} URL${batch.length === 1 ? '' : 's'})`);
    if (text) console.log(text);
    if (status !== 200 && status !== 202) process.exit(1);
  }
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
