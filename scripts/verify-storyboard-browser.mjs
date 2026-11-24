import { execFileSync, spawn } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const grid = require('../web/guides/storyboard-grid.js');
const { encodePng, tutorialGrid, smallGrid } = require('./storyboard-grid-fixture.cjs');
const root = path.resolve(import.meta.dirname, '..', 'web');
const work = fs.mkdtempSync('/tmp/storyboard-grid-');
const downloads = path.join(work, 'downloads');
fs.mkdirSync(downloads);
const shots = path.resolve('/opt/cursor/artifacts');
fs.mkdirSync(shots, { recursive: true });

const tutorial = tutorialGrid();
const small = smallGrid();
const wide = encodePng(new Uint8Array(8193 * 4).fill(255), 8193, 1);
const crowded = encodePng(new Uint8Array(11 * 10 * 4).fill(255), 11, 10);
fs.writeFileSync(path.join(work, 'tutorial.png'), tutorial.png);
fs.writeFileSync(path.join(work, 'small.png'), small.png);
fs.writeFileSync(path.join(work, 'wide.png'), wide);
fs.writeFileSync(path.join(work, 'crowded.png'), crowded);
execFileSync('ffmpeg', [
  '-y', '-f', 'lavfi', '-i', 'color=c=red:s=64x36:d=0.5',
  '-f', 'lavfi', '-i', 'color=c=blue:s=64x36:d=0.5',
  '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0',
  '-pix_fmt', 'yuv420p', path.join(work, 'clip.mp4')
], { stdio: 'ignore' });

function sha256(bytes) {
  return crypto.createHash('sha256').update(Buffer.from(bytes)).digest('hex');
}

const expected = Object.fromEntries(tutorial.frames.map(frame => [frame.index, sha256(tutorial.crops[frame.index])]));
const smallExpected = Object.fromEntries(small.frames.map(frame => [frame.index, sha256(small.crops[frame.index])]));

function serve() {
  const server = http.createServer((request, response) => {
    const url = new URL(request.url, 'http://127.0.0.1');
    const target = path.normalize(path.join(root, decodeURIComponent(url.pathname)));
    if (!target.startsWith(root) || !fs.existsSync(target) || fs.statSync(target).isDirectory()) {
      response.writeHead(404);
      response.end('missing');
      return;
    }
    const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png' };
    response.writeHead(200, { 'content-type': types[path.extname(target)] || 'application/octet-stream' });
    response.end(fs.readFileSync(target));
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

async function waitFor(predicate, label, timeout = 15000) {
  const start = Date.now();
  let last;
  while (Date.now() - start < timeout) {
    last = await predicate();
    if (last) return last;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error('Timed out waiting for ' + label + ': ' + JSON.stringify(last));
}

function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let id = 0;
  const pending = new Map();
  const events = [];
  return new Promise((resolve, reject) => {
    ws.addEventListener('open', () => resolve({
      send(method, params = {}, sessionId) {
        const messageId = ++id;
        return new Promise((done, fail) => {
          pending.set(messageId, { done, fail });
          ws.send(JSON.stringify({ id: messageId, method, params, sessionId }));
        });
      },
      events,
      close() { ws.close(); }
    }));
    ws.addEventListener('error', reject);
    ws.addEventListener('message', event => {
      const message = JSON.parse(event.data);
      if (message.id && pending.has(message.id)) {
        const { done, fail } = pending.get(message.id);
        pending.delete(message.id);
        if (message.error) fail(new Error(JSON.stringify(message.error)));
        else done(message.result);
      } else events.push(message);
    });
  });
}

async function main() {
  const server = await serve();
  const port = server.address().port;
  const chromePort = 9333;
  const profile = path.join(work, 'chrome');
  const chrome = spawn('google-chrome', [
    '--headless=new', '--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage',
    '--remote-allow-origins=*', '--remote-debugging-port=' + chromePort,
    '--user-data-dir=' + profile, '--no-first-run', 'about:blank'
  ], { stdio: 'ignore' });
  try {
    const version = await waitFor(async () => {
      try {
        const response = await fetch('http://127.0.0.1:' + chromePort + '/json/version');
        return response.ok ? response.json() : null;
      } catch { return null; }
    }, 'Chrome DevTools');
    const browser = await connect(version.webSocketDebuggerUrl);
    const { targetId } = await browser.send('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await browser.send('Target.attachToTarget', { targetId, flatten: true });
    const send = (method, params) => browser.send(method, params, sessionId);
    await send('Page.enable');
    await send('Runtime.enable');
    await send('DOM.enable');
    await send('Emulation.setDeviceMetricsOverride', { width: 1100, height: 1500, deviceScaleFactor: 1, mobile: false });
    try {
      await browser.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: downloads, eventsEnabled: true });
    } catch (error) {
      await send('Page.setDownloadBehavior', { behavior: 'allow', downloadPath: downloads });
    }

    async function evaluate(expression) {
      const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
      if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
      return result.result.value;
    }
    async function capture(name, selector) {
      await evaluate('document.querySelector(' + JSON.stringify(selector) + ').scrollIntoView({ block: "start" })');
      await new Promise(resolve => setTimeout(resolve, 150));
      fs.writeFileSync(path.join(shots, name), Buffer.from((await send('Page.captureScreenshot', { format: 'png' })).data, 'base64'));
    }
    async function statusText() {
      return evaluate('document.getElementById("grid-status").textContent');
    }
    async function upload(filePath, selector = '#grid-file') {
      const documentNode = await send('DOM.getDocument');
      const { nodeId } = await send('DOM.querySelector', { nodeId: documentNode.root.nodeId, selector });
      await send('DOM.setFileInputFiles', { nodeId, files: [filePath] });
    }
    function emptyDownloads() {
      for (const name of fs.readdirSync(downloads)) fs.rmSync(path.join(downloads, name), { force: true });
    }
    async function newestDownload() {
      const names = fs.readdirSync(downloads).filter(name => !name.endsWith('.crdownload'));
      return names.length ? path.join(downloads, names[0]) : '';
    }

    await send('Page.navigate', { url: 'http://127.0.0.1:' + port + '/guides/storyboard-grid.html' });
    await waitFor(() => evaluate('Boolean(window.StoryboardGrid && document.getElementById("grid-download"))'), 'storyboard page');
    await evaluate(`
      for (const [id, value] of [['grid-rows','2'],['grid-columns','3'],['grid-padding','20'],['grid-gap-x','12'],['grid-gap-y','12']]) {
        const input = document.getElementById(id);
        input.value = value;
        input.dispatchEvent(new Event('input', { bubbles: true }));
      }
    `);
    await upload(path.join(work, 'tutorial.png'));
    const ready = await waitFor(async () => {
      const text = await statusText();
      return text.includes('Each panel is 320×180') ? text : '';
    }, 'tutorial crop');
    if (!ready.includes('Source 1024×412') || !ready.includes('Remaining 960×360')) {
      throw new Error('Status does not report the tutorial geometry: ' + ready);
    }
    const preview = await evaluate(`(async () => {
      const expected = ${JSON.stringify(expected)};
      async function hex(data) {
        const digest = await crypto.subtle.digest('SHA-256', data);
        return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
      }
      async function hashCanvas(canvas, x, y, width, height) {
        const image = canvas.getContext('2d', { willReadFrequently: true }).getImageData(x, y, width, height);
        return hex(image.data);
      }
      const source = document.getElementById('grid-source');
      const boxes = [...document.querySelectorAll('.grid-box')].map(box => ({
        text: box.textContent, name: box.dataset.name, x: Number(box.dataset.x), y: Number(box.dataset.y),
        w: Number(box.dataset.w), h: Number(box.dataset.h), pressed: box.getAttribute('aria-pressed')
      }));
      const cards = [];
      for (const figure of document.querySelectorAll('#grid-frames figure')) {
        const canvas = figure.querySelector('canvas');
        cards.push({
          index: Number(figure.dataset.index), name: figure.dataset.name, caption: figure.textContent.trim(),
          width: canvas.width, height: canvas.height, checked: figure.querySelector('input').checked,
          hash: await hashCanvas(canvas, 0, 0, canvas.width, canvas.height)
        });
      }
      const sourceHashes = {};
      for (const box of boxes) sourceHashes[box.text] = await hashCanvas(source, box.x, box.y, box.w, box.h);
      const gap = source.getContext('2d').getImageData(340, 20, 1, 1).data;
      const download = document.getElementById('grid-download').textContent;
      return { boxes, cards, sourceHashes, gap: [...gap], download, sourceWidth: source.width, sourceHeight: source.height };
    })()`);
    if (preview.sourceWidth !== 1024 || preview.sourceHeight !== 412) throw new Error('Source canvas size ' + preview.sourceWidth + 'x' + preview.sourceHeight);
    if (preview.gap.join(',') !== '255,0,255,255') throw new Error('Gap pixel is not the marker: ' + preview.gap);
    if (preview.boxes.map(box => box.text).join(',') !== '1,2,3,4,5,6') throw new Error('Box numbers ' + JSON.stringify(preview.boxes));
    for (const frame of tutorial.frames) {
      const box = preview.boxes[frame.index - 1];
      if (box.x !== frame.x || box.y !== frame.y || box.w !== frame.width || box.h !== frame.height || box.name !== frame.name || box.pressed !== 'true') {
        throw new Error('Box does not match plan: ' + JSON.stringify(box));
      }
      const card = preview.cards.find(item => item.index === frame.index);
      if (!card || card.width !== 320 || card.height !== 180 || !card.checked || !card.caption.includes('320×180') || !card.caption.includes(frame.name)) {
        throw new Error('Card does not match plan: ' + JSON.stringify(card));
      }
      if (card.hash !== preview.sourceHashes[String(frame.index)]) {
        throw new Error('Panel ' + frame.index + ' preview does not match its crop');
      }
      if (frame.index !== 1 && card.hash !== expected[frame.index]) {
        throw new Error('Opaque panel ' + frame.index + ' does not match the file pixels');
      }
    }
    const alphaDiff = await evaluate(`(async () => {
      const expected = Uint8Array.from(atob(${JSON.stringify(Buffer.from(tutorial.crops[1]).toString('base64'))}), char => char.charCodeAt(0));
      const data = document.querySelector('#grid-frames figure[data-index="1"] canvas').getContext('2d').getImageData(0, 0, 320, 180).data;
      const samples = [];
      for (let i = 0; i < data.length; i += 4) {
        if (data[i] !== expected[i] || data[i + 1] !== expected[i + 1] || data[i + 2] !== expected[i + 2] || data[i + 3] !== expected[i + 3]) {
          samples.push({ pixel: i / 4, got: [data[i], data[i + 1], data[i + 2], data[i + 3]], want: [expected[i], expected[i + 1], expected[i + 2], expected[i + 3]] });
        }
      }
      return samples;
    })()`);
    if (alphaDiff.length > 1) throw new Error('Panel 1 changed more than the known transparent pixel: ' + JSON.stringify(alphaDiff));
    if (alphaDiff.length === 1) {
      const sample = alphaDiff[0];
      const rgbDrift = Math.max(Math.abs(sample.got[0] - sample.want[0]), Math.abs(sample.got[1] - sample.want[1]), Math.abs(sample.got[2] - sample.want[2]));
      if (sample.want[3] !== 128 || sample.got[3] !== 128 || rgbDrift > 1) {
        throw new Error('Unexpected transparent-pixel change: ' + JSON.stringify(sample));
      }
    }
    if (preview.download !== 'Download ZIP (6 frames)') throw new Error('Download label ' + preview.download);
    await capture('storyboard-grid-tutorial.png', '#grid-stage');

    emptyDownloads();
    await evaluate(`
      for (const index of [2, 4, 5]) {
        const input = document.querySelector('#grid-frames figure[data-index="' + index + '"] input');
        input.checked = false;
        input.dispatchEvent(new Event('change', { bubbles: true }));
      }
    `);
    const subsetLabel = await evaluate('document.getElementById("grid-download").textContent');
    if (subsetLabel !== 'Download ZIP (3 frames)') throw new Error('Subset label ' + subsetLabel);
    await evaluate('document.getElementById("grid-download").click()');
    const zipPath = await waitFor(newestDownload, 'ZIP download');
    const unzipped = grid.readZip(fs.readFileSync(zipPath));
    if (unzipped.map(file => file.name).join(',') !== 'shot_01.png,shot_03.png,shot_06.png') {
      throw new Error('ZIP names ' + unzipped.map(file => file.name).join(','));
    }
    const zipPixels = await evaluate(`(async () => {
      const files = ${JSON.stringify(unzipped.map(file => ({ name: file.name, bytes: [...file.bytes] })))};
      async function hex(data) {
        const digest = await crypto.subtle.digest('SHA-256', data);
        return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
      }
      const out = [];
      for (const file of files) {
        const blob = new Blob([new Uint8Array(file.bytes)], { type: 'image/png' });
        const image = new Image();
        image.src = URL.createObjectURL(blob);
        await image.decode();
        const canvas = document.createElement('canvas');
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        const context = canvas.getContext('2d');
        context.drawImage(image, 0, 0);
        out.push({ name: file.name, width: canvas.width, height: canvas.height, hash: await hex(context.getImageData(0, 0, canvas.width, canvas.height).data) });
        URL.revokeObjectURL(image.src);
      }
      return out;
    })()`);
    const cardHashes = Object.fromEntries(preview.cards.map(card => [card.name, card.hash]));
    const zipExpected = { 'shot_01.png': cardHashes['shot_01.png'], 'shot_03.png': expected[3], 'shot_06.png': expected[6] };
    for (const file of zipPixels) {
      if (file.width !== 320 || file.height !== 180 || file.hash !== zipExpected[file.name] || file.hash !== cardHashes[file.name]) {
        throw new Error('ZIP PNG pixels differ from the preview: ' + JSON.stringify(file));
      }
    }

    emptyDownloads();
    await evaluate(`
      document.getElementById('grid-none').click();
      for (const figure of document.querySelectorAll('#grid-frames figure')) {
        const input = figure.querySelector('input');
        const wanted = figure.dataset.index === '4';
        if (input.checked !== wanted) {
          input.checked = wanted;
          input.dispatchEvent(new Event('change', { bubbles: true }));
        }
      }
    `);
    const singleLabel = await evaluate('document.getElementById("grid-download").textContent');
    if (singleLabel !== 'Download shot_04.png') throw new Error('Single label ' + singleLabel);
    await evaluate('document.getElementById("grid-download").click()');
    const pngPath = await waitFor(newestDownload, 'single PNG download');
    if (path.basename(pngPath) !== 'shot_04.png') throw new Error('Single filename ' + path.basename(pngPath));
    const single = fs.readFileSync(pngPath);
    if (single[0] !== 137 || single.toString('ascii', 1, 4) !== 'PNG') throw new Error('Single download is not a PNG');
    const singleHash = await evaluate(`(async () => {
      const image = new Image();
      image.src = 'data:image/png;base64,${single.toString('base64')}';
      await image.decode();
      const canvas = document.createElement('canvas');
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext('2d');
      context.drawImage(image, 0, 0);
      const digest = await crypto.subtle.digest('SHA-256', context.getImageData(0, 0, canvas.width, canvas.height).data);
      return { width: canvas.width, height: canvas.height, hash: [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('') };
    })()`);
    if (singleHash.width !== 320 || singleHash.height !== 180 || singleHash.hash !== expected[4] || singleHash.hash !== cardHashes['shot_04.png']) {
      throw new Error('shot_04.png pixels differ: ' + JSON.stringify(singleHash));
    }

    await evaluate(`
      document.getElementById('grid-gap-x').value = '13';
      document.getElementById('grid-gap-x').dispatchEvent(new Event('input', { bubbles: true }));
    `);
    const invalid = await waitFor(async () => {
      const text = await statusText();
      return text.includes('1 px left over') ? text : '';
    }, 'leftover error');
    if (!invalid.includes('Nothing was cropped') || !invalid.includes('does not round') || !invalid.includes('958 px')) {
      throw new Error('Leftover message is incomplete: ' + invalid);
    }
    const cleared = await evaluate('document.querySelectorAll("#grid-frames canvas").length + ":" + document.getElementById("grid-download").disabled + ":" + document.querySelectorAll(".grid-box").length');
    if (cleared !== '0:true:0') throw new Error('Invalid geometry still exposes crops: ' + cleared);
    await capture('storyboard-grid-invalid.png', '#grid-status');
    emptyDownloads();
    await evaluate('document.getElementById("grid-download").click()');
    await new Promise(resolve => setTimeout(resolve, 400));
    if (fs.readdirSync(downloads).length) throw new Error('Invalid geometry downloaded a file');

    await evaluate(`
      for (const [id, value] of [['grid-rows','0'],['grid-columns','3'],['grid-padding','-3'],['grid-gap-x','1.5'],['grid-gap-y',' 2']]) {
        document.getElementById(id).value = value;
      }
      document.getElementById('grid-rows').dispatchEvent(new Event('input', { bubbles: true }));
    `);
    const rejected = await statusText();
    for (const phrase of ['Nothing was cropped', 'Rows must be at least 1', 'Outer padding must be a whole number', 'Horizontal gap must be a whole number', 'Vertical gap must be a whole number']) {
      if (!rejected.includes(phrase)) throw new Error('Missing "' + phrase + '" in: ' + rejected);
    }

    await evaluate(`
      for (const [id, value] of [['grid-rows','2'],['grid-columns','3'],['grid-padding','20'],['grid-gap-x','12'],['grid-gap-y','12']]) {
        document.getElementById(id).value = value;
      }
      document.getElementById('grid-rows').dispatchEvent(new Event('input', { bubbles: true }));
    `);
    await waitFor(async () => (await statusText()).includes('Each panel is 320×180'), 'restored tutorial crop');

    const beforeReplace = fs.readdirSync(downloads).length;
    await evaluate(`
      const transfer = new DataTransfer();
      transfer.items.add(new File([new Uint8Array([1, 2, 3, 4])], 'broken.png', { type: 'image/png' }));
      const input = document.getElementById('grid-file');
      input.files = transfer.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    `);
    const broken = await waitFor(async () => {
      const text = await statusText();
      return text.includes('could not decode') ? text : '';
    }, 'broken image');
    if (!broken.includes('Nothing was cropped')) throw new Error(broken);
    const afterBroken = await evaluate('document.querySelectorAll("#grid-frames canvas").length + ":" + document.getElementById("grid-download").disabled');
    if (afterBroken !== '0:true') throw new Error('Broken replacement kept a download: ' + afterBroken);
    emptyDownloads();
    await evaluate('document.getElementById("grid-download").click()');
    await new Promise(resolve => setTimeout(resolve, 400));
    if (fs.readdirSync(downloads).length !== beforeReplace && fs.readdirSync(downloads).length !== 0) {
      throw new Error('Broken image still downloaded');
    }

    await evaluate(`
      for (const [id, value] of [['grid-rows','2'],['grid-columns','2'],['grid-padding','1'],['grid-gap-x','2'],['grid-gap-y','2']]) {
        document.getElementById(id).value = value;
      }
    `);
    await upload(path.join(work, 'small.png'));
    await waitFor(async () => (await statusText()).includes('Each panel is 3×2'), 'replacement grid');
    const replacement = await evaluate(`(async () => {
      const expected = ${JSON.stringify(smallExpected)};
      async function hex(canvas) {
        const digest = await crypto.subtle.digest('SHA-256', canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data);
        return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
      }
      const cards = [];
      for (const figure of document.querySelectorAll('#grid-frames figure')) {
        const canvas = figure.querySelector('canvas');
        cards.push({ index: Number(figure.dataset.index), width: canvas.width, height: canvas.height, hash: await hex(canvas) });
      }
      return cards;
    })()`);
    if (replacement.length !== 4) throw new Error('Replacement panel count ' + replacement.length);
    for (const card of replacement) {
      if (card.width !== 3 || card.height !== 2 || card.hash !== smallExpected[card.index] || card.hash === expected[card.index]) {
        throw new Error('Replacement card kept or missed pixels: ' + JSON.stringify(card));
      }
    }

    await evaluate('document.getElementById("grid-clear").click()');
    const clearedStatus = await statusText();
    if (!clearedStatus.includes('previous image can no longer be downloaded')) throw new Error(clearedStatus);
    const clearedState = await evaluate('document.querySelectorAll("#grid-frames canvas").length + ":" + document.getElementById("grid-stage").hidden + ":" + document.getElementById("grid-download").disabled');
    if (clearedState !== '0:true:true') throw new Error('Clear left export state: ' + clearedState);
    emptyDownloads();
    await evaluate('document.getElementById("grid-download").click()');
    await new Promise(resolve => setTimeout(resolve, 400));
    if (fs.readdirSync(downloads).length) throw new Error('Clear still downloaded the previous image');

    await upload(path.join(work, 'wide.png'));
    const wideStatus = await waitFor(async () => {
      const text = await statusText();
      return text.includes('8193×1') ? text : '';
    }, 'wide image refusal');
    if (!wideStatus.includes('Nothing was cropped') || !wideStatus.includes('8192')) throw new Error(wideStatus);

    await evaluate(`
      for (const [id, value] of [['grid-rows','10'],['grid-columns','11'],['grid-padding','0'],['grid-gap-x','0'],['grid-gap-y','0']]) {
        document.getElementById(id).value = value;
      }
    `);
    await upload(path.join(work, 'crowded.png'));
    const crowdedStatus = await waitFor(async () => {
      const text = await statusText();
      return text.includes('110 panels') ? text : '';
    }, 'panel limit');
    if (!crowdedStatus.includes('Nothing was cropped')) throw new Error(crowdedStatus);

    await evaluate(`
      for (const [id, value] of [['grid-rows','2'],['grid-columns','3'],['grid-padding','20'],['grid-gap-x','12'],['grid-gap-y','12']]) {
        document.getElementById(id).value = value;
      }
    `);
    await upload(path.join(work, 'tutorial.png'));
    await waitFor(async () => (await statusText()).includes('Each panel is 320×180'), 'tutorial before JPEG');
    const jpeg = await evaluate(`(async () => {
      async function hex(data) {
        const digest = await crypto.subtle.digest('SHA-256', data);
        return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
      }
      const source = document.getElementById('grid-source');
      const lossless = await hex(source.getContext('2d').getImageData(20, 20, 320, 180).data);
      const blob = await new Promise(resolve => source.toBlob(resolve, 'image/jpeg', 0.92));
      const transfer = new DataTransfer();
      transfer.items.add(new File([blob], 'grid.jpg', { type: 'image/jpeg' }));
      const input = document.getElementById('grid-file');
      input.files = transfer.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
      return { lossless, bytes: blob.size };
    })()`);
    await waitFor(async () => (await statusText()).includes('Each panel is 320×180'), 'JPEG crop');
    const jpegCompare = await evaluate(`(async () => {
      async function hex(data) {
        const digest = await crypto.subtle.digest('SHA-256', data);
        return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
      }
      const source = document.getElementById('grid-source');
      const card = document.querySelector('#grid-frames figure[data-index="1"] canvas');
      return {
        source: await hex(source.getContext('2d').getImageData(20, 20, 320, 180).data),
        card: await hex(card.getContext('2d').getImageData(0, 0, card.width, card.height).data),
        width: card.width, height: card.height
      };
    })()`);
    if (jpegCompare.width !== 320 || jpegCompare.height !== 180) throw new Error('JPEG crop changed size ' + JSON.stringify(jpegCompare));
    if (jpegCompare.source !== jpegCompare.card) throw new Error('JPEG preview does not match the decoded crop');
    if (jpegCompare.card === jpeg.lossless) throw new Error('JPEG crop unexpectedly matched the lossless PNG pixels');

    await send('Page.navigate', { url: 'http://127.0.0.1:' + port + '/guides/first-last-frame-video.html' });
    await waitFor(() => evaluate('Boolean(document.getElementById("first-download"))'), 'frame page');
    const frameInitial = await evaluate(`({
      status: document.getElementById('frame-status').textContent,
      firstDisabled: document.getElementById('first-download').disabled,
      lastDisabled: document.getElementById('last-download').disabled,
      first: [document.getElementById('first-canvas').width, document.getElementById('first-canvas').height],
      last: [document.getElementById('last-canvas').width, document.getElementById('last-canvas').height],
      linked: Boolean(document.querySelector('a[href="storyboard-grid.html"]')),
      accept: document.getElementById('video-file').accept
    })`);
    if (!frameInitial.status.includes('Nothing is uploaded') || !frameInitial.firstDisabled || !frameInitial.lastDisabled) {
      throw new Error('Frame extractor initial state changed: ' + JSON.stringify(frameInitial));
    }
    if (frameInitial.first.join('x') !== '16x9' || frameInitial.last.join('x') !== '16x9' || frameInitial.accept !== 'video/*' || !frameInitial.linked) {
      throw new Error('Frame extractor markup changed: ' + JSON.stringify(frameInitial));
    }
    await upload(path.join(work, 'clip.mp4'), '#video-file');
    const frameStatus = await waitFor(async () => {
      const text = await evaluate('document.getElementById("frame-status").textContent');
      if (text.includes('Decoded the first frame') || text.includes('could not') || text.includes('did not') || text.includes('too long') || text.includes('duration')) return text;
      return '';
    }, 'frame extractor', 20000);
    if (!frameStatus.includes('Decoded the first frame')) throw new Error('Frame extractor failed on a local clip: ' + frameStatus);
    const enabled = await evaluate('!document.getElementById("first-download").disabled && !document.getElementById("last-download").disabled');
    if (!enabled) throw new Error('Frame download buttons stayed disabled: ' + frameStatus);
    const colors = await evaluate(`({
      first: [...document.getElementById('first-canvas').getContext('2d').getImageData(32, 18, 1, 1).data],
      last: [...document.getElementById('last-canvas').getContext('2d').getImageData(32, 18, 1, 1).data],
      size: [document.getElementById('first-canvas').width, document.getElementById('first-canvas').height]
    })`);
    if (colors.size.join('x') !== '64x36') throw new Error('Frame canvas size ' + colors.size);
    if (!(colors.first[0] > colors.first[2] && colors.first[0] > 150)) throw new Error('First frame is not the red frame: ' + colors.first);
    if (!(colors.last[2] > colors.last[0] && colors.last[2] > 150)) throw new Error('Last frame is not the blue frame: ' + colors.last);
    await capture('first-last-frame-regression.png', '#frame-status');

    console.log(JSON.stringify({
      ok: true,
      tutorial: '1024x412 padding 20 gap 12 -> 320x180; opaque panels match the file and every preview matches its download',
      zip: unzipped.map(file => file.name),
      single: 'shot_04.png',
      invalid: 'gap 13 reports 958 px with 1 px left over and downloads nothing',
      replacement: 'broken file and a 10x8 grid replace the previous pixels',
      clear: 'download stays disabled',
      jpeg: 'decoded JPEG crop matches its preview and differs from the PNG',
      frameExtractor: frameStatus,
      frameColors: colors
    }, null, 2));
    browser.close();
  } finally {
    chrome.kill('SIGKILL');
    server.close();
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
