const minimaxKey = process.env.MINIMAX_API_KEY;
const deepseekKey = process.env.DEEPSEEK_API_KEY;

if (!minimaxKey) throw new Error('MINIMAX_API_KEY is not configured.');
if (!deepseekKey) throw new Error('DEEPSEEK_API_KEY is not configured.');

const TIMEOUT_MS = 90_000;

function safeJson(text) {
  const cleaned = String(text || '')
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();
  try {
    return JSON.parse(cleaned);
  } catch {
    const start = cleaned.indexOf('{');
    const end = cleaned.lastIndexOf('}');
    if (start >= 0 && end > start) return JSON.parse(cleaned.slice(start, end + 1));
    throw new Error('Model did not return parseable JSON.');
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function fetchWithTimeout(url, options, label) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);
  const started = Date.now();
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const raw = await response.text();
    let body;
    try {
      body = JSON.parse(raw);
    } catch {
      body = { raw: raw.slice(0, 300) };
    }
    if (!response.ok) {
      const providerMessage = body?.error?.message || body?.base_resp?.status_msg || body?.raw || 'Unknown provider error';
      throw new Error(`${label} HTTP ${response.status}: ${String(providerMessage).slice(0, 240)}`);
    }
    return { body, elapsedMs: Date.now() - started };
  } catch (error) {
    if (error?.name === 'AbortError') throw new Error(`${label} timed out after ${TIMEOUT_MS / 1000}s.`);
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function listMiniMaxModels(baseUrl) {
  const { body, elapsedMs } = await fetchWithTimeout(`${baseUrl}/models`, {
    headers: { Authorization: `Bearer ${minimaxKey}` },
  }, `MiniMax model list (${baseUrl})`);
  const ids = Array.isArray(body.data) ? body.data.map(model => model.id).filter(Boolean) : [];
  console.log(`[MiniMax] ${baseUrl} model list OK in ${elapsedMs}ms; M3 listed=${ids.includes('MiniMax-M3')}.`);
  return ids;
}

async function callMiniMax(baseUrl, useResponseFormat = true) {
  const payload = {
    model: 'MiniMax-M3',
    messages: [
      { role: 'system', content: 'You are a multilingual technical editor. Return only the requested JSON object, with no markdown and no reasoning text.' },
      { role: 'user', content: 'Return exactly one JSON object with keys english, zh, ja, html. Explain in one short sentence per language that a favicon identifies a website in browser tabs. html must be one valid <p> element in English. Do not add other keys.' },
    ],
    temperature: 0.2,
    max_tokens: 900,
    reasoning_split: true,
    ...(useResponseFormat ? { response_format: { type: 'json_object' } } : {}),
  };
  return fetchWithTimeout(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${minimaxKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  }, `MiniMax M3 chat (${baseUrl})`);
}

async function probeMiniMax() {
  const bases = ['https://api.minimax.io/v1', 'https://api.minimaxi.com/v1'];
  const errors = [];
  for (const baseUrl of bases) {
    try {
      const ids = await listMiniMaxModels(baseUrl);
      if (!ids.includes('MiniMax-M3')) {
        console.log(`[MiniMax] M3 is not advertised by ${baseUrl}; attempting a direct entitlement check.`);
      }
      let result;
      try {
        result = await callMiniMax(baseUrl, true);
      } catch (error) {
        if (!/response_format|unsupported|unknown field|invalid.*parameter/i.test(error.message)) throw error;
        console.log('[MiniMax] response_format is unsupported; retrying with prompt-enforced JSON.');
        result = await callMiniMax(baseUrl, false);
      }
      const content = result.body?.choices?.[0]?.message?.content || '';
      assert(content && !/<think>|reasoning:/i.test(content), 'MiniMax final content contains reasoning spill or is empty.');
      const data = safeJson(content);
      assert(Object.keys(data).sort().join(',') === ['english', 'html', 'ja', 'zh'].sort().join(','), 'MiniMax JSON keys do not match the contract.');
      assert(typeof data.english === 'string' && data.english.length > 20, 'MiniMax English output is too short.');
      assert(/[\u3400-\u9fff]/u.test(data.zh || ''), 'MiniMax Chinese output is missing.');
      assert(/[\u3040-\u30ff]/u.test(data.ja || ''), 'MiniMax Japanese output is missing.');
      assert(/^<p>[\s\S]*<\/p>$/.test(data.html || ''), 'MiniMax HTML contract failed.');
      const usage = result.body?.usage || {};
      console.log(`[MiniMax] PASS via ${baseUrl}; ${result.elapsedMs}ms; prompt=${usage.prompt_tokens ?? usage.input_tokens ?? '?'} completion=${usage.completion_tokens ?? usage.output_tokens ?? '?'}.`);
      return;
    } catch (error) {
      errors.push(`${baseUrl}: ${error.message}`);
      console.warn(`[MiniMax] ${baseUrl} failed: ${error.message}`);
    }
  }
  throw new Error(`MiniMax M3 probe failed on both official regions. ${errors.join(' | ')}`);
}

async function callDeepSeek(messages, { thinking, effort, maxTokens }) {
  const payload = {
    model: 'deepseek-v4-flash',
    messages,
    thinking: { type: thinking },
    ...(effort ? { reasoning_effort: effort } : {}),
    max_tokens: maxTokens,
    response_format: { type: 'json_object' },
  };
  return fetchWithTimeout('https://api.deepseek.com/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${deepseekKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  }, `DeepSeek V4 Flash (${thinking})`);
}

async function probeDeepSeek() {
  const jsonResult = await callDeepSeek([
    { role: 'system', content: 'Return valid JSON only.' },
    { role: 'user', content: 'Return exactly {"provider":"deepseek","mode":"non-thinking","ok":true}.' },
  ], { thinking: 'disabled', maxTokens: 300 });
  const jsonData = safeJson(jsonResult.body?.choices?.[0]?.message?.content || '');
  assert(jsonData.provider === 'deepseek' && jsonData.mode === 'non-thinking' && jsonData.ok === true, 'DeepSeek strict JSON probe failed.');
  const jsonUsage = jsonResult.body?.usage || {};
  console.log(`[DeepSeek] strict JSON PASS; ${jsonResult.elapsedMs}ms; prompt=${jsonUsage.prompt_tokens ?? '?'} completion=${jsonUsage.completion_tokens ?? '?'}.`);

  const auditResult = await callDeepSeek([
    { role: 'system', content: 'You are a strict fact checker. Use only the supplied evidence and return JSON only.' },
    { role: 'user', content: 'Evidence: The HTML Standard defines rel="icon" as a link type for an icon representing the current page. Claim: A favicon can represent the current page. Return exactly {"verdict":"pass|fail","unsupportedClaims":[],"reason":"..."}.' },
  ], { thinking: 'enabled', effort: 'low', maxTokens: 900 });
  const auditData = safeJson(auditResult.body?.choices?.[0]?.message?.content || '');
  assert(auditData.verdict === 'pass', `DeepSeek audit verdict was ${auditData.verdict || 'missing'}.`);
  assert(Array.isArray(auditData.unsupportedClaims) && auditData.unsupportedClaims.length === 0, 'DeepSeek audit returned unsupported claims.');
  const auditUsage = auditResult.body?.usage || {};
  console.log(`[DeepSeek] source audit PASS; ${auditResult.elapsedMs}ms; prompt=${auditUsage.prompt_tokens ?? '?'} completion=${auditUsage.completion_tokens ?? '?'} reasoning=${auditUsage.completion_tokens_details?.reasoning_tokens ?? '?'}.`);
}

console.log('Starting isolated provider probe. No repository content will be generated or published.');
await probeMiniMax();
await probeDeepSeek();
console.log('ALL PROVIDER PROBES PASSED.');

