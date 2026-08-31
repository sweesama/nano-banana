import assert from 'node:assert/strict';

import {
  MODELS,
  VERIFIER_MODELS,
  classifyModelError,
  extractRelevantSourceText,
  findAbsoluteProductClaim,
  isAuthoritativeExternalSource,
  isRepairableContentError,
  modelsForItem,
  hasDanglingDescriptionEnding,
  normalizeDescription,
  normalizeTitle,
  parseModelList,
  parseModelRoute,
  sanitizeHtml,
  validateArticle,
  verifierModelsForAuthor,
  resolveMaxTokens,
} from './generate-article.js';

assert.equal(MODELS.includes('stepfun-ai/step-3.7-flash'), false);
assert.equal(VERIFIER_MODELS.includes('stepfun-ai/step-3.7-flash'), false);
assert.equal(MODELS.length, 2);
assert.equal(VERIFIER_MODELS.length, 2);
assert.equal(MODELS[0], 'minimax:MiniMax-M3');
assert.equal(MODELS[1], 'deepseek:deepseek-v4-flash');
assert.equal(VERIFIER_MODELS[0], 'deepseek:deepseek-v4-flash');
assert.equal(modelsForItem({ category: 'Benchmarks' })[0], MODELS[0]);
assert.equal(modelsForItem({ category: 'API Tutorial' })[0], MODELS[0]);
assert.deepEqual(parseModelList('model/a, model/b, model/a', ['fallback']), ['model/a', 'model/b']);
assert.deepEqual(parseModelRoute('minimax:MiniMax-M3'), { provider: 'minimax', model: 'MiniMax-M3', routeName: 'minimax:MiniMax-M3' });
assert.equal(verifierModelsForAuthor('minimax:MiniMax-M3')[0], 'deepseek:deepseek-v4-flash');
assert.equal(verifierModelsForAuthor('deepseek:deepseek-v4-flash')[0], 'minimax:MiniMax-M3');
assert.equal(resolveMaxTokens('minimax', 'article', 7000), 24576);
assert.equal(resolveMaxTokens('minimax', 'source-audit', 1800), 8192);
assert.equal(resolveMaxTokens('deepseek', 'source-audit', 1800), 4096);
assert.equal(resolveMaxTokens('nvidia', 'article', 7000), 7000);
assert.equal(classifyModelError({ status: 410, message: 'Gone' }), 'permanent');
assert.equal(classifyModelError({ status: 429, message: 'Rate limited' }), 'transient');
assert.equal(classifyModelError({ status: 401, message: 'Unauthorized' }), 'auth');
assert.equal(classifyModelError({ code: 'PROVIDER_UNCONFIGURED' }), 'unconfigured');
assert.equal(findAbsoluteProductClaim('<p>Always validate the response before saving it.</p>'), '');
assert.equal(findAbsoluteProductClaim('<p>The API always works in every region.</p>').toLowerCase(), 'always works');
assert.equal(findAbsoluteProductClaim('<p>Usage is not guaranteed and quotas may change.</p>'), '');
assert.equal(findAbsoluteProductClaim('<p>Guaranteed uptime is included.</p>').toLowerCase(), 'guaranteed uptime');
assert.equal(isRepairableContentError(new Error('Article does not cite every required source near the relevant claim.')), true);
assert.equal(isRepairableContentError(new Error('Unsafe HTML detected.')), false);

const longSource = `${'Introductory navigation text. '.repeat(500)} ${'Unrelated model notes. '.repeat(500)} Current image editing example: client.interactions.create uses input objects with type image, base64 data, and mime_type. The response_format object controls aspect_ratio. Generated images include SynthID. Breaking changes require the current request shape.`;
const relevantSource = extractRelevantSourceText(longSource, 'Gemini image editing API input image workflow');
assert.ok(relevantSource.length <= 12000);
assert.match(relevantSource, /client\.interactions\.create/);
assert.match(relevantSource, /response_format/);
assert.match(relevantSource, /SynthID/);

const longDescription = 'This source-backed guide explains how image benchmark Elo scores work, what uncertainty means, and why a single leaderboard position should never be treated as permanent proof of model quality or universal superiority.';
const normalizedDescription = normalizeDescription(longDescription);
assert.ok(normalizedDescription.length <= 170);
assert.match(normalizedDescription, /[.!?]$/);

const incompleteDescription = 'A practical tutorial on the Gemini image API input image workflow in Python, covering base64 uploads, multi-turn editing with previous_interaction_id, and.';
const repairedDescription = normalizeDescription(incompleteDescription);
assert.equal(repairedDescription, 'A practical tutorial on the Gemini image API input image workflow in Python, covering base64 uploads, multi-turn editing with previous_interaction_id.');
assert.equal(hasDanglingDescriptionEnding(repairedDescription), false);
assert.equal(normalizeDescription('A'.repeat(170)).length, 170);
assert.equal(normalizeDescription(''), '');

const normalizedTitle = normalizeTitle(
  'A very long introduction before the required phrase AI image benchmark and several unnecessary trailing promises for every reader',
  'AI image benchmark',
);
assert.ok(normalizedTitle.length <= 70);
assert.match(normalizedTitle.toLowerCase(), /ai image benchmark/);

const sourceUrl = 'https://ai.google.dev/gemini-api/docs/image-generation';
const internalOne = 'https://www.nano-banana.live/faq.html';
const internalTwo = 'https://www.nano-banana.live/guides/quickstart.html';
const item = {
  sourceUrls: [sourceUrl, internalOne, internalTwo],
};
const cluster = {
  primaryTerms: ['Gemini image API'],
  internalLinks: [internalOne, internalTwo],
};

assert.equal(isAuthoritativeExternalSource(sourceUrl), true);
assert.equal(isAuthoritativeExternalSource(internalOne), false);

const sanitized = sanitizeHtml('<p>Read <a href="https://example.com">the source</a>.</p><pre><code><link rel="icon" href="/favicon.ico"></code></pre>');
assert.match(sanitized, /<\/a>/);
assert.match(sanitized, /target="_blank" rel="noopener"/);
assert.match(sanitized, /&lt;link rel="icon"/);
assert.doesNotMatch(sanitized, /<link rel="icon"/);

const baseArticle = {
  title: 'Gemini image API deployment boundary explained',
  description: 'A source-backed explanation of hosted image inference, local clients, and separately released open-weight models for practical deployment decisions.',
  content: `<h2>Cloud route</h2><p>Read <a href="${sourceUrl}">Google's documentation</a>, <a href="${internalOne}">the FAQ</a>, and <a href="${internalTwo}">the quickstart</a>.</p><h2>Decision</h2><ul><li>Privacy</li><li>Cost</li><li>Hardware</li></ul>`,
};

assert.doesNotThrow(() => validateArticle(baseArticle, item, cluster));
assert.throws(
  () => validateArticle({ ...baseArticle, content: `${baseArticle.content}<a href="#">broken</a>` }, item, cluster),
  /Placeholder link/,
);
assert.throws(
  () => validateArticle({ ...baseArticle, content: `${baseArticle.content}<p>Use the free tier.</p>` }, item, cluster),
  /pricing page/,
);
assert.throws(
  () => validateArticle({ ...baseArticle, description: `${baseArticle.description.slice(0, -1)}, and.` }, item, cluster),
  /incomplete phrase/,
);

console.log('Publishing guard tests passed.');
