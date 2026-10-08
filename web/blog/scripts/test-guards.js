import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

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
  generateFromQueue,
  recordQueueFailure,
  saveDiagnostic,
  selectQueueItem,
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
assert.equal(classifyModelError({ code: 'PROVIDER_UNAVAILABLE' }), 'unconfigured');
assert.equal(findAbsoluteProductClaim('<p>Always validate the response before saving it.</p>'), '');
assert.equal(findAbsoluteProductClaim('<p>The API always works in every region.</p>').toLowerCase(), 'always works');
assert.equal(findAbsoluteProductClaim('<p>Usage is not guaranteed and quotas may change.</p>'), '');
assert.equal(findAbsoluteProductClaim('<p>Guaranteed uptime is included.</p>').toLowerCase(), 'guaranteed uptime');
for (const text of [
  'RunPod does not offer guaranteed uptime.',
  'There is no guaranteed uptime.',
  'Do not assume guaranteed availability.',
  'There is no evidence of guaranteed quality.',
  'Guaranteed uptime is not included.',
  'The service does not always work.',
  'It doesn&#39;t provide guaranteed uptime.',
]) {
  assert.equal(findAbsoluteProductClaim('<p>' + text + '</p>'), '', text);
}
for (const text of [
  'Not only guaranteed uptime but also useful monitoring.',
  'No guaranteed uptime, but guaranteed quality is included.',
  'Do not assume guaranteed uptime. This service has guaranteed uptime.',
  '<p>No documented guarantee is offered.</p><p>Guaranteed uptime is included.</p>',
  'Guaranteed uptime is not limited to enterprise plans.',
]) {
  assert.notEqual(findAbsoluteProductClaim(text), '', text);
}
assert.equal(classifyModelError(new SyntaxError('Bad control character in string literal')), 'retryable-output');
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

const now = new Date('2026-10-08T00:00:00Z');
const failedTopic = { slug: 'failed-topic', status: 'pending' };
recordQueueFailure(failedTopic, new Error('Article content needs revision.'), now);
assert.equal(failedTopic.failureCount, 1);
assert.equal(failedTopic.nextAttemptAt, '2026-10-09T00:00:00.000Z');
assert.equal(selectQueueItem([failedTopic], new Set(), now), undefined);
const freshTopic = { slug: 'fresh-topic', status: 'pending' };
assert.equal(selectQueueItem([failedTopic, freshTopic], new Set(), new Date('2026-10-10')), freshTopic);
assert.equal(selectQueueItem([failedTopic, freshTopic], new Set(['fresh-topic']), new Date('2026-10-10')), failedTopic);
recordQueueFailure(failedTopic, new Error('Still invalid.'), now);
recordQueueFailure(failedTopic, new Error('Still invalid.'), now);
assert.equal(failedTopic.status, 'needs-review');
assert.equal(failedTopic.nextAttemptAt, undefined);
assert.equal(selectQueueItem([failedTopic], new Set(), new Date('2026-10-20')), undefined);

const diagnosticsDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nano-banana-blog-tests-'));
try {
  const recordPath = saveDiagnostic(failedTopic, 'validation', {
    article: baseArticle,
    error: Object.assign(new Error('Source audit failed.'), { audit: { verdict: 'fail' } }),
  }, diagnosticsDir);
  const saved = JSON.parse(fs.readFileSync(recordPath, 'utf8'));
  assert.equal(saved.article.content, baseArticle.content);
  assert.equal(saved.error.audit.verdict, 'fail');
  assert.equal(saved.topic.slug, failedTopic.slug);

  const research = { clusters: [{ ...cluster, categories: ['API Tutorial'] }] };
  const makeTopic = slug => ({ slug, category: 'API Tutorial', depth: 'standard', status: 'pending', sourceUrls: item.sourceUrls });
  const queue = [makeTopic('bad'), makeTopic('good')];
  const attempts = [];
  const generated = await generateFromQueue(queue, [], research, {
    now, diagnosticsDir,
    generate: async topic => {
      attempts.push(topic.slug);
      if (topic.slug === 'bad') throw new Error('Absolute product claim detected.');
      return baseArticle;
    },
  });
  assert.deepEqual(attempts, ['bad', 'good']);
  assert.equal(generated.item.slug, 'good');
  assert.equal(queue[0].failureCount, 1);
  assert.equal(queue[0].status, 'pending');

  const badQueue = [makeTopic('bad-one'), makeTopic('bad-two'), makeTopic('not-attempted')];
  await assert.rejects(generateFromQueue(badQueue, [], research, {
    now, diagnosticsDir, generate: async () => { throw new Error('Source audit failed.'); },
  }), /Source audit failed/);
  assert.deepEqual(badQueue.map(topic => topic.failureCount || 0), [1, 1, 0]);

  const outageQueue = [makeTopic('provider-outage'), makeTopic('not-attempted')];
  await assert.rejects(generateFromQueue(outageQueue, [], research, {
    now, diagnosticsDir, generate: async () => { throw Object.assign(new Error('Unauthorized'), { status: 401 }); },
  }), /Unauthorized/);
  assert.ok(outageQueue.every(topic => !topic.failureCount && topic.status === 'pending'));
  await assert.rejects(generateFromQueue(outageQueue, [], research, {
    now, diagnosticsDir, generate: async () => { throw Object.assign(new Error('All verifier models were unavailable.'), { code: 'PROVIDER_UNAVAILABLE' }); },
  }), /unavailable/);
  assert.ok(outageQueue.every(topic => !topic.failureCount));
  assert.equal(await generateFromQueue([failedTopic], [], research, { now, diagnosticsDir }), null);
} finally {
  fs.rmSync(diagnosticsDir, { recursive: true, force: true });
}

console.log('Publishing guard and queue recovery tests passed.');
