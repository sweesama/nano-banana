const fs = require('fs');
const path = require('path');

const API_KEY = process.env.AA_API_KEY;
const ROOT = path.resolve(__dirname, '..');
const HTML_PATH = path.join(ROOT, 'web', 'benchmarks', 'index.html');
const API_BASE = 'https://artificialanalysis.ai/api/v2/media';

if (!API_KEY) {
  console.error('Missing AA_API_KEY. Create an Artificial Analysis API key and expose it only as a server-side secret.');
  process.exit(1);
}

function escapeHtml(value) {
  return String(value ?? '—')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatCi(value) {
  if (value === null || value === undefined || value === '') return '—';
  const number = Number(value);
  return Number.isFinite(number) ? `±${number}` : escapeHtml(value);
}

function creatorDomain(creator) {
  const domains = {
    OpenAI: 'openai.com',
    Google: 'google.com',
    'Black Forest Labs': 'blackforestlabs.ai',
    'Black Forest': 'blackforestlabs.ai',
    ByteDance: 'seed.bytedance.com',
    Bytedance: 'seed.bytedance.com',
    Reve: 'reve-art.com',
    Sourceful: 'riverflow.ai',
    Tencent: 'tencent.com',
    Alibaba: 'alibaba.com',
    xAI: 'x.ai',
    NVIDIA: 'nvidia.com',
    'Microsoft AI': 'microsoft.com',
    HiDream: 'hidream.ai',
    Recraft: 'recraft.ai',
  };
  return domains[creator] || '';
}

function creatorCell(creator) {
  const safeCreator = escapeHtml(creator || 'Unknown');
  const domain = creatorDomain(creator);
  const icon = domain
    ? `<img src="https://www.google.com/s2/favicons?domain=${domain}&sz=32" class="creator-icon" alt="${safeCreator}" onerror="this.style.display='none'">`
    : '';
  return `${icon}${safeCreator}`;
}

function rowClass(rank) {
  if (rank === 1) return 'rank-1';
  if (rank === 2) return 'rank-2';
  if (rank === 3) return 'rank-3';
  return '';
}

function textRows(data) {
  return data.map((item, index) => {
    const rank = Number(item.rank) || index + 1;
    const creator = item.model_creator?.name || 'Unknown';
    return `              <tr class="${rowClass(rank)}">
                <td style="font-weight:${rank <= 3 ? 800 : 700};color:${rank === 1 ? '#ffd700' : rank === 2 ? '#e5e7eb' : rank === 3 ? '#fb923c' : '#d1d5db'};font-size:1.1rem;">${rank}</td>
                <td style="color:#9ca3af;">—</td>
                <td>${creatorCell(creator)}</td>
                <td style="font-weight:600;">${escapeHtml(item.name)}</td>
                <td style="text-align:center;font-weight:${rank <= 3 ? 700 : 600};color:#fff;font-size:1.05rem;">${Number(item.elo).toLocaleString()}</td>
                <td style="text-align:center;color:#6b7280;font-size:0.85rem;">${formatCi(item.ci_95)}</td>
                <td style="text-align:center;color:#9ca3af;">—</td>
                <td style="text-align:center;color:#9ca3af;font-size:0.85rem;">—</td>
                <td style="text-align:right;color:#9ca3af;font-size:0.8rem;">—</td>
              </tr>`;
  }).join('\n');
}

function editingRows(data) {
  return data.map((item, index) => {
    const rank = Number(item.rank) || index + 1;
    const creator = item.model_creator?.name || 'Unknown';
    return `              <tr class="${rowClass(rank)}">
                <td style="font-weight:${rank <= 3 ? 800 : 700};color:${rank === 1 ? '#ffd700' : rank === 2 ? '#e5e7eb' : rank === 3 ? '#fb923c' : '#4ade80'};font-size:1.1rem;">${rank}</td>
                <td style="color:#9ca3af;">—</td>
                <td>${creatorCell(creator)}</td>
                <td style="font-weight:600;">${escapeHtml(item.name)}</td>
                <td style="text-align:center;font-weight:${rank <= 3 ? 700 : 600};color:${rank <= 3 ? '#fff' : '#4ade80'};font-size:1.05rem;">${Number(item.elo).toLocaleString()}</td>
                <td style="text-align:center;color:#6b7280;font-size:0.85rem;">${formatCi(item.ci_95)}</td>
                <td style="text-align:center;color:#9ca3af;font-size:0.85rem;">—</td>
                <td style="text-align:right;color:#9ca3af;font-size:0.8rem;">—</td>
              </tr>`;
  }).join('\n');
}

function replaceTbody(html, sectionId, rows) {
  const sectionStart = html.indexOf(`id="${sectionId}"`);
  if (sectionStart < 0) throw new Error(`Section not found: ${sectionId}`);
  const tbodyStart = html.indexOf('<tbody>', sectionStart);
  const tbodyEnd = html.indexOf('</tbody>', tbodyStart);
  if (tbodyStart < 0 || tbodyEnd < 0) throw new Error(`Table body not found: ${sectionId}`);
  return `${html.slice(0, tbodyStart)}<tbody>\n${rows}\n            ${html.slice(tbodyEnd)}`;
}

async function fetchLeaderboard(endpoint) {
  const response = await fetch(`${API_BASE}/${endpoint}/models/free`, {
    headers: { 'x-api-key': API_KEY, accept: 'application/json' },
  });
  if (!response.ok) throw new Error(`${endpoint} request failed with HTTP ${response.status}`);
  const payload = await response.json();
  if (!Array.isArray(payload.data) || payload.data.length === 0) throw new Error(`${endpoint} returned no models`);
  return payload.data.slice(0, 10);
}

async function main() {
  const [textToImage, imageEditing] = await Promise.all([
    fetchLeaderboard('text-to-image'),
    fetchLeaderboard('image-editing'),
  ]);
  const refreshed = new Intl.DateTimeFormat('en-US', { year: 'numeric', month: 'long', day: 'numeric', timeZone: 'UTC' }).format(new Date());
  let html = fs.readFileSync(HTML_PATH, 'utf8');
  html = replaceTbody(html, 'text-to-image', textRows(textToImage));
  html = replaceTbody(html, 'image-editing', editingRows(imageEditing));
  html = html.replace(/Static leaderboard snapshot sourced from/g, 'Automated leaderboard snapshot sourced from');
  html = html.replace(/Snapshot date shown below\./g, 'The snapshot is refreshed automatically when the scheduled sync succeeds.');
  html = html.replace(/Data refreshed: <strong>.*?<\/strong>/, `Data refreshed: <strong>${refreshed}</strong>`);
  html = html.replace(/Updated April \d{1,2}, \d{4}/, `Updated ${refreshed}`);
  fs.writeFileSync(HTML_PATH, html, 'utf8');
  console.log(`Updated ${textToImage.length} text-to-image and ${imageEditing.length} image-editing models.`);
  console.log(`Snapshot date: ${refreshed}`);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
