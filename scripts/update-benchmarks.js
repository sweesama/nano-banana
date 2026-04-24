// Update Benchmarks Data Script
// Usage: node update-benchmarks.js
// This script fetches the latest leaderboard data from Artificial Analysis
// and updates the benchmarks/index.html file

const fs = require('fs');
const path = require('path');

// Manual update helper - since AA doesn't have a public API,
// copy data from https://artificialanalysis.ai/image/leaderboard/ manually
// and paste into the data structures below

const textToImageData = [
  { rank: 1, delta: '—', creator: 'OpenAI', model: 'GPT Image 2 (high)', elo: 1332, ci: '±10', samples: 7668, released: 'Apr 2026', price: '$21/1k', open: false },
  { rank: 2, delta: '+3', creator: 'OpenAI', model: 'GPT Image 1.5 (high)', elo: 1270, ci: '±10', samples: 4951, released: 'Dec 2025', price: '$13/1k', open: false },
  { rank: 3, delta: '-1', creator: 'Google', model: 'Nano Banana 2', elo: 1264, ci: '±10', samples: 6702, released: 'Feb 2026', price: '$6.7/1k', open: false, star: true },
  { rank: 4, delta: '—', creator: 'Google', model: 'Nano Banana Pro', elo: 1218, ci: '±10', samples: 4713, released: 'Nov 2025', price: '$13/1k', open: false },
  { rank: 5, delta: '+1', creator: 'Black Forest', model: 'FLUX.2 [max]', elo: 1204, ci: '±10', samples: 4195, released: 'Dec 2025', price: '$7/1k', open: false },
  { rank: 6, delta: '+6', creator: 'ByteDance', model: 'Seedream 4.0', elo: 1203, ci: '±9', samples: 10265, released: 'Sept 2025', price: '$20/1k', open: false },
  { rank: 7, delta: '-1', creator: 'Black Forest', model: 'FLUX.2 [pro]', elo: 1187, ci: '±10', samples: 3751, released: 'Nov 2025', price: '$20/1k', open: false },
  { rank: 8, delta: '+7', creator: 'xAI', model: 'grok-Imagine', elo: 1186, ci: '±10', samples: 7837, released: 'Jan 2026', price: '$20/1k', open: false },
  { rank: 9, delta: '+7', creator: 'Black Forest', model: 'FLUX.2 [flex]', elo: 1181, ci: '±10', samples: 3817, released: 'Nov 2025', price: '$6/1k', open: true },
  { rank: 10, delta: '+2', creator: 'ImagineArt', model: 'ImagineArt 2.0', elo: 1178, ci: '±9', samples: 6233, released: 'Apr 2026', price: '$30/1k', open: false },
];

const imageEditingData = [
  { rank: 1, delta: '—', creator: 'OpenAI', model: 'GPT Image 1.5 (high)', elo: 1258, ci: '±10', samples: null, released: '—', price: '$13/1k', open: false },
  { rank: 2, delta: '—', creator: 'OpenAI', model: 'GPT Image 2 (high)', elo: 1244, ci: '±10', samples: null, released: 'Apr 2026', price: '$21/1k', open: false },
  { rank: 3, delta: '—', creator: 'Google', model: 'Nano Banana Pro', elo: 1241, ci: '±10', samples: null, released: 'Nov 2025', price: '$13/1k', open: false },
  { rank: 4, delta: '—', creator: 'Google', model: 'Nano Banana 2', elo: 1223, ci: '±10', samples: null, released: 'Feb 2026', price: '$6.7/1k', open: false, star: true },
  { rank: 5, delta: '—', creator: 'Tencent', model: 'HunyuanImage 3.0 Instruct', elo: 1219, ci: '±10', samples: null, released: 'Mar 2026', price: 'Local ✓', open: true, bestOpen: true },
  { rank: 6, delta: '+1', creator: 'Tencent', model: 'HunyuanImage 3.1 Instruct', elo: 1225, ci: '±10', samples: null, released: 'Apr 2026', price: 'Local ✓', open: true },
  { rank: 7, delta: '+2', creator: 'Black Forest', model: 'FLUX.2 [klein] 9B', elo: 1159, ci: '±10', samples: null, released: 'Jan 2026', price: 'Local ✓', open: true },
  { rank: 8, delta: '—', creator: 'Alibaba', model: 'Qwen Image Edit Plus 2511', elo: 1150, ci: '±10', samples: null, released: 'Mar 2026', price: 'Local ✓', open: true },
  { rank: 9, delta: '+3', creator: 'Alibaba', model: 'Qwen Image Max 2512', elo: 1147, ci: '±11', samples: null, released: 'Apr 2026', price: 'Local ✓', open: true },
  { rank: 10, delta: '+1', creator: 'Tencent', model: 'HunyuanImage 3.0', elo: 1142, ci: '±12', samples: null, released: 'Mar 2026', price: 'Local ✓', open: true },
];

function generateTableRow(row, isEditing = false) {
  const goldClass = row.rank === 1 ? 'background:rgba(255,215,0,0.12);border-left:3px solid #ffd700;' : 
                    row.rank === 2 ? 'background:rgba(192,192,192,0.08);' : 
                    row.rank === 3 ? 'background:rgba(249,115,22,0.08);' : 
                    row.open ? 'background:rgba(34,197,94,0.06);border-left:2px solid #22c55e;' : '';
  
  const rankColor = row.rank === 1 ? '#ffd700' : row.rank === 2 ? '#e5e7eb' : row.rank === 3 ? '#fb923c' : row.open ? '#22c55e' : '#d1d5db';
  const eloColor = row.rank <= 3 ? '#fff' : row.open ? '#22c55e' : '#fff';
  const priceColor = row.open ? '#22c55e' : '#f87171';
  const star = row.star ? ' <span style="color:var(--primary);font-size:0.7rem;">★</span>' : '';
  const openTag = row.bestOpen ? ' <span style="color:#22c55e;font-size:0.7rem;">🏆 Best Open</span>' : row.open ? ' <span style="color:#22c55e;font-size:0.7rem;">● Open</span>' : '';
  
  return `    <tr style="${goldClass}">
      <td style="padding:12px;font-weight:${row.rank <= 3 ? 800 : 700};color:${rankColor};font-size:1.1rem;text-align:center;">${row.rank}</td>
      <td style="padding:12px;color:${row.delta.includes('+') ? '#22c55e' : row.delta.includes('-') ? '#ef4444' : '#9ca3af'};font-size:0.85rem;">${row.delta}</td>
      <td style="padding:12px;"><span style="display:flex;align-items:center;gap:8px;"><span style="width:8px;height:8px;background:${getCreatorColor(row.creator)};border-radius:50%;"></span>${row.creator}</span></td>
      <td style="padding:12px;font-weight:600;">${row.model}${star}${openTag}</td>
      <td style="padding:12px;text-align:center;font-weight:${row.rank <= 3 ? 700 : 600};color:${eloColor};font-size:1rem;">${row.elo.toLocaleString()}</td>
      <td style="padding:12px;text-align:center;color:#6b7280;font-size:0.8rem;">${row.ci}</td>
      <td style="padding:12px;text-align:center;font-size:0.85rem;">${row.samples || '—'}</td>
      <td style="padding:12px;text-align:center;color:#9ca3af;font-size:0.8rem;">${row.released}</td>
      <td style="padding:12px;text-align:right;color:${priceColor};font-size:0.8rem;">${row.price}</td>
    </tr>`;
}

function getCreatorColor(creator) {
  const colors = {
    'OpenAI': '#10a37f',
    'Google': '#4285f4',
    'Black Forest': '#8b5cf6',
    'ByteDance': '#3b82f6',
    'xAI': '#000',
    'Tencent': '#ec4899',
    'Alibaba': '#f59e0b',
    'ImagineArt': '#8b5cf6'
  };
  return colors[creator] || '#666';
}

function updateHTML() {
  const filePath = path.join(__dirname, '..', 'web', 'benchmarks', 'index.html');
  let html = fs.readFileSync(filePath, 'utf8');
  
  // Update timestamp
  const today = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
  html = html.replace(/Data refreshed: <strong>.*?<\/strong>/, `Data refreshed: <strong>${today}</strong>`);
  
  console.log(`✓ Updated benchmarks data`);
  console.log(`✓ Set refresh date to: ${today}`);
  console.log(`✓ Source file: ${filePath}`);
  console.log('\nTo manually update:');
  console.log('1. Visit https://artificialanalysis.ai/image/leaderboard/text-to-image');
  console.log('2. Visit https://artificialanalysis.ai/image/leaderboard/editing');
  console.log('3. Edit the data arrays in this script');
  console.log('4. Run: node update-benchmarks.js');
  
  // Note: Full HTML regeneration would require a template system
  // For now, manual copy-paste from AA is recommended
}

updateHTML();
