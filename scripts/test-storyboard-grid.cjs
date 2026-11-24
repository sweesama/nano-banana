const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const grid = require('../web/guides/storyboard-grid.js');
const { MARKER, decodePng, tutorialGrid, smallGrid } = require('./storyboard-grid-fixture.cjs');

function pixel(rgba, width, x, y) {
  const offset = (y * width + x) * 4;
  return [rgba[offset], rgba[offset + 1], rgba[offset + 2], rgba[offset + 3]];
}

function assertBytes(actual, expected) {
  assert.equal(actual.length, expected.length);
  assert.equal(Buffer.from(actual).equals(Buffer.from(expected)), true);
}

function assertNoMarker(bytes) {
  for (let offset = 0; offset < bytes.length; offset += 4) {
    if (bytes[offset] === MARKER[0] && bytes[offset + 1] === MARKER[1] && bytes[offset + 2] === MARKER[2] && bytes[offset + 3] === MARKER[3]) {
      assert.fail('crop contains the padding/gap marker at byte ' + offset);
    }
  }
}

const tutorial = tutorialGrid();
assert.deepEqual(decodePng(tutorial.png), { width: 1024, height: 412, rgba: tutorial.rgba });
const tutorialPlan = grid.planGrid({ width: 1024, height: 412, ...tutorial.settings });
assert.equal(tutorialPlan.ok, true);
assert.equal(tutorialPlan.cellWidth, 320);
assert.equal(tutorialPlan.cellHeight, 180);
assert.equal(tutorialPlan.remainingWidth, 960);
assert.equal(tutorialPlan.remainingHeight, 360);
assert.deepEqual(tutorialPlan.frames, tutorial.frames);
const tutorialCrops = grid.buildCrops({ data: tutorial.rgba, width: 1024, height: 412 }, tutorialPlan);
assert.equal(tutorialCrops.ok, true);
for (const frame of tutorial.frames) {
  assertBytes(tutorialCrops.crops[frame.index], tutorial.crops[frame.index]);
  assert.equal(tutorialCrops.crops[frame.index].length, 320 * 180 * 4);
  assertNoMarker(tutorialCrops.crops[frame.index]);
  assert.deepEqual(pixel(tutorialCrops.crops[frame.index], 320, 0, 0), [frame.index, 20, 30, 255]);
  assert.deepEqual(pixel(tutorialCrops.crops[frame.index], 320, 1, 0), [frame.index, 40, 50, 255]);
}
assert.deepEqual(pixel(tutorial.rgba, 1024, 0, 0), MARKER);
assert.deepEqual(pixel(tutorial.rgba, 1024, 339, 20), tutorial.colors[0]);
assert.deepEqual(pixel(tutorial.rgba, 1024, 340, 20), MARKER);
assert.deepEqual(pixel(tutorial.rgba, 1024, 351, 20), MARKER);
assert.deepEqual(pixel(tutorial.rgba, 1024, 352, 20), [2, 20, 30, 255]);
assert.deepEqual(pixel(tutorial.rgba, 1024, 20, 199), tutorial.colors[0]);
assert.deepEqual(pixel(tutorial.rgba, 1024, 20, 200), MARKER);
assert.deepEqual(pixel(tutorial.rgba, 1024, 20, 211), MARKER);
assert.deepEqual(pixel(tutorial.rgba, 1024, 20, 212), [4, 20, 30, 255]);
assert.deepEqual(pixel(tutorial.rgba, 1024, 1003, 391), tutorial.colors[5]);
assert.deepEqual(pixel(tutorial.rgba, 1024, 1004, 391), MARKER);
assert.deepEqual(pixel(tutorial.rgba, 1024, 20, 392), MARKER);
assert.deepEqual(pixel(tutorialCrops.crops[1], 320, 10, 10), [9, 8, 7, 128]);

const small = smallGrid();
const smallPlan = grid.planGrid({ width: 10, height: 8, ...small.settings });
assert.equal(smallPlan.ok, true);
assert.deepEqual(smallPlan.frames, small.frames);
assert.equal(smallPlan.cellWidth, 3);
assert.equal(smallPlan.cellHeight, 2);
const smallCrops = grid.buildCrops({ data: small.rgba, width: 10, height: 8 }, smallPlan);
assert.equal(smallCrops.ok, true);
for (const frame of small.frames) {
  assertBytes(smallCrops.crops[frame.index], small.crops[frame.index]);
  assertNoMarker(smallCrops.crops[frame.index]);
}
assert.deepEqual(pixel(small.rgba, 10, 0, 0), MARKER);
assert.deepEqual(pixel(small.rgba, 10, 4, 1), MARKER);
assert.deepEqual(pixel(small.rgba, 10, 5, 1), MARKER);
assert.deepEqual(pixel(small.rgba, 10, 1, 3), MARKER);
assert.deepEqual(pixel(small.rgba, 10, 9, 7), MARKER);

const session = grid.createSession();
const first = session.replaceImage();
const firstCommit = session.commit(first.generation, smallPlan, smallCrops.crops);
assert.equal(firstCommit.ok, true);
const selected = session.exportSelection(first.generation, firstCommit.revision, [4, 1, 2]);
assert.equal(selected.ok, true);
assert.equal(selected.kind, 'zip');
assert.equal(selected.name, 'storyboard-shots.zip');
assert.deepEqual(selected.frames.map(frame => frame.name), ['shot_01.png', 'shot_02.png', 'shot_04.png']);
assertBytes(selected.frames[0].bytes, small.crops[1]);
assertBytes(selected.frames[2].bytes, small.crops[4]);
selected.frames[0].bytes[0] = 123;
const again = session.exportSelection(first.generation, firstCommit.revision, [1]);
assert.equal(again.kind, 'png');
assert.equal(again.name, 'shot_01.png');
assert.equal(again.frames[0].bytes[0], 1);
smallCrops.crops[1][0] = 77;
assert.equal(session.exportSelection(first.generation, firstCommit.revision, [1]).frames[0].bytes[0], 1);

const replacementPixels = new Uint8Array(4 * 4 * 4);
replacementPixels.fill(0);
replacementPixels[0] = 4;
replacementPixels[1] = 5;
replacementPixels[2] = 6;
replacementPixels[3] = 255;
const replacementPlan = grid.planGrid({ width: 4, height: 4, rows: 1, columns: 1, padding: 0, gapX: 9, gapY: 0 });
assert.equal(replacementPlan.ok, true);
assert.deepEqual(replacementPlan.notes, ['Horizontal gap is unused because there is one column.']);
const replacementCrops = grid.buildCrops({ data: replacementPixels, width: 4, height: 4 }, replacementPlan);
const second = session.replaceImage();
assert.equal(session.commit(first.generation, smallPlan, smallCrops.crops).ok, false);
assert.equal(session.exportSelection(first.generation, firstCommit.revision, [1]).ok, false);
const secondCommit = session.commit(second.generation, replacementPlan, replacementCrops.crops);
const replaced = session.exportSelection(second.generation, secondCommit.revision, [1]);
assert.equal(replaced.kind, 'png');
assert.equal(replaced.frames[0].name, 'shot_01.png');
assert.deepEqual(pixel(replaced.frames[0].bytes, 4, 0, 0), [4, 5, 6, 255]);
assert.notDeepEqual(pixel(replaced.frames[0].bytes, 4, 0, 0), pixel(small.crops[1], 3, 0, 0));
session.clear();
assert.equal(session.exportSelection(second.generation, secondCommit.revision, [1]).ok, false);
assert.match(session.exportSelection(second.generation, secondCommit.revision, [1]).error, /Nothing was downloaded|before downloading/);

const fresh = grid.createSession();
const freshId = fresh.replaceImage();
const freshCommit = fresh.commit(freshId.generation, smallPlan, smallCrops.crops);
assert.equal(fresh.exportSelection(freshId.generation, freshCommit.revision, []).error.includes('at least one'), true);
assert.equal(fresh.commit(freshId.generation, { ok: false, frames: [] }, {}).ok, false);
assert.equal(fresh.exportSelection(freshId.generation, freshCommit.revision, [1]).ok, false);

const outside = grid.buildCrops(
  { data: small.rgba, width: 10, height: 8 },
  { ok: true, width: 10, height: 8, frames: [small.frames[0], { index: 2, x: 9, y: 0, width: 2, height: 2, name: 'shot_02.png' }] }
);
assert.equal(outside.ok, false);
assert.equal(outside.crops, undefined);
assert.match(outside.error, /outside the image/);
assert.match(grid.cropRgba({ data: small.rgba, width: 10, height: 8 }, { index: 1, x: -1, y: 0, width: 1, height: 1 }).error, /outside/);

function errorsOf(input) {
  const result = grid.planGrid(input);
  assert.equal(result.ok, false);
  assert.deepEqual(result.frames, []);
  return result.errors.join(' ');
}
assert.match(errorsOf({ width: 10, height: 10, rows: 1, columns: 3, padding: 0, gapX: 0, gapY: 0 }), /remaining width is 10 px/);
assert.match(errorsOf({ width: 10, height: 10, rows: 1, columns: 3, padding: 0, gapX: 0, gapY: 0 }), /1 px left over/);
assert.match(errorsOf({ width: 10, height: 10, rows: 1, columns: 3, padding: 0, gapX: 0, gapY: 0 }), /does not round/);
assert.match(errorsOf({ width: 1024, height: 412, rows: 2, columns: 3, padding: 20, gapX: 13, gapY: 12 }), /958 px/);
assert.match(errorsOf({ width: 1024, height: 412, rows: 2, columns: 3, padding: 20, gapX: 13, gapY: 12 }), /1 px left over/);
assert.doesNotMatch(errorsOf({ width: 10, height: 10, rows: 3, columns: 1, padding: 0, gapX: 0, gapY: 0 }), /remaining width/);
assert.match(errorsOf({ width: 10, height: 10, rows: 3, columns: 1, padding: 0, gapX: 0, gapY: 0 }), /remaining height is 10 px/);
assert.match(errorsOf({ width: 10, height: 10, rows: 1, columns: 1, padding: 5, gapX: 0, gapY: 0 }), /remaining width is 0 px/);
assert.match(errorsOf({ width: 10, height: 10, rows: 1, columns: 1, padding: 6, gapX: 0, gapY: 0 }), /remaining width is -2 px/);
assert.match(errorsOf({ width: 10, height: 10, rows: 0, columns: 1, padding: 0, gapX: 0, gapY: 0 }), /Rows must be at least 1/);
assert.match(errorsOf({ width: 10, height: 10, rows: -2, columns: 1, padding: 0, gapX: 0, gapY: 0 }), /Rows must be at least 1/);
assert.match(errorsOf({ width: 10, height: 10, rows: 1.5, columns: 1, padding: 0, gapX: 0, gapY: 0 }), /Rows must be a whole number/);
assert.match(errorsOf({ width: 10, height: 10, rows: 1, columns: 1, padding: -1, gapX: 0, gapY: 0 }), /Outer padding must be zero or greater/);
assert.match(errorsOf({ width: 10, height: 10, rows: 1, columns: 1, padding: 0, gapX: -4, gapY: 1.2 }), /Horizontal gap must be zero or greater/);
assert.match(errorsOf({ width: 10, height: 10, rows: 1, columns: 1, padding: 0, gapX: -4, gapY: 1.2 }), /Vertical gap must be a whole number/);
assert.match(errorsOf({ width: 0, height: 10, rows: 1, columns: 1, padding: 0, gapX: 0, gapY: 0 }), /Image width must be a positive whole number/);
assert.match(errorsOf({ width: 9000, height: 10, rows: 1, columns: 1, padding: 0, gapX: 0, gapY: 0 }), /stops above 8192/);
assert.match(errorsOf({ width: 100, height: 100, rows: 10, columns: 11, padding: 0, gapX: 0, gapY: 0 }), /110 panels is above the 100-panel limit/);
assert.equal(grid.planGrid({ width: 30, height: 30, rows: 10, columns: 10, padding: 0, gapX: 0, gapY: 0 }).ok, true);

const blank = grid.parseSettings(null);
assert.equal(blank.ok, false);
assert.equal(blank.settings, undefined);
assert.equal(blank.errors.length, 5);
assert.match(grid.parseSettings({ rows: '0', columns: '1', padding: '0', gapX: '0', gapY: '0' }).errors.join(' '), /Rows must be at least 1/);
assert.match(grid.parseSettings({ rows: '2.5', columns: '3', padding: '0', gapX: '0', gapY: '0' }).errors.join(' '), /Rows must be a whole number/);
assert.match(grid.parseSettings({ rows: '-1', columns: '3', padding: '0', gapX: '0', gapY: '0' }).errors.join(' '), /Rows must be a whole number/);
assert.match(grid.parseSettings({ rows: ' 2', columns: '3', padding: '0', gapX: '0', gapY: '0' }).errors.join(' '), /Rows must be a whole number/);
assert.match(grid.parseSettings({ rows: '2', columns: '3', padding: '01', gapX: '0', gapY: '0' }).errors.join(' '), /Outer padding must be a whole number/);
const parsed = grid.parseSettings({ rows: '2', columns: '3', padding: '20', gapX: '12', gapY: '12' });
assert.deepEqual(parsed.settings, tutorial.settings);

assert.equal(grid.crc32(new TextEncoder().encode('123456789')), 0xCBF43926);
const zipped = grid.zipStore([
  { name: 'shot_01.png', bytes: small.png },
  { name: 'shot_03.png', bytes: tutorial.crops[3] }
]);
const view = new DataView(zipped.buffer, zipped.byteOffset, zipped.byteLength);
assert.equal(view.getUint16(12, true), grid.ZIP_DOS_DATE);
const unzipped = grid.readZip(zipped);
assert.deepEqual(unzipped.map(file => file.name), ['shot_01.png', 'shot_03.png']);
assertBytes(unzipped[0].bytes, small.png);
assertBytes(unzipped[1].bytes, tutorial.crops[3]);
const tampered = new Uint8Array(zipped);
tampered[30 + 'shot_01.png'.length] ^= 0xff;
assert.throws(() => grid.readZip(tampered), /checksum/);
assert.throws(() => grid.zipStore([]), /ZIP needs/);
assert.throws(() => grid.zipStore([{ name: 'shot_01.png', bytes: small.png }, { name: 'shot_01.png', bytes: small.png }]), /Duplicate/);
assert.throws(() => grid.zipStore([{ name: '../shot_01.png', bytes: small.png }]), /plain file name/);

const page = fs.readFileSync(path.join(__dirname, '../web/guides/storyboard-grid.html'), 'utf8');
const toolScript = page.match(/<script src="storyboard-grid.js"><\/script>\s*<script>([\s\S]*?)<\/script>/)[1];
assert.equal(toolScript.includes('fetch('), false);
assert.equal(toolScript.includes('XMLHttpRequest'), false);
assert.equal(toolScript.includes('WebSocket'), false);
assert.deepEqual(toolScript.match(/drawImage\(/g), ['drawImage(']);
assert.match(toolScript, /drawImage\(image, 0, 0\)/);
assert.match(toolScript, /buildCrops/);
assert.match(toolScript, /putImageData/);
assert.match(toolScript, /zipStore/);
assert.match(page, /8192/);
assert.match(page, /100 panels/);
assert.match(page, /32 million pixels/);
assert.equal(grid.MAX_EDGE, 8192);
assert.equal(grid.MAX_PANELS, 100);
assert.equal(grid.MAX_PIXELS, 32000000);

const framePage = fs.readFileSync(path.join(__dirname, '../web/guides/first-last-frame-video.html'), 'utf8');
const frameScript = [...framePage.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(match => match[1]).find(source => source.includes('first-frame.png'));
assert.equal(crypto.createHash('sha256').update(frameScript).digest('hex'), 'f734807926930a038ce538df5c0965751198034ed0b5931bdb8b64ae6ab1e7d6');
assert.match(frameScript, /last-frame\.png/);
assert.match(frameScript, /Nothing is uploaded/);
assert.equal(frameScript.includes('shot_01'), false);
assert.equal(frameScript.includes('StoryboardGrid'), false);
assert.match(framePage, /accept="video\/\*"/);
assert.match(framePage, /href="storyboard-grid.html"/);

console.log('Storyboard grid tests passed: tutorial 1024x412 -> 320x180, padding and gaps, subset order, ZIP bytes, invalid geometry, and frame-extractor script unchanged.');
