const zlib = require('node:zlib');
const { crc32 } = require('../web/guides/storyboard-grid.js');

const MARKER = [255, 0, 255, 255];
const TUTORIAL_FRAMES = [
  { index: 1, row: 0, column: 0, x: 20, y: 20, width: 320, height: 180, name: 'shot_01.png' },
  { index: 2, row: 0, column: 1, x: 352, y: 20, width: 320, height: 180, name: 'shot_02.png' },
  { index: 3, row: 0, column: 2, x: 684, y: 20, width: 320, height: 180, name: 'shot_03.png' },
  { index: 4, row: 1, column: 0, x: 20, y: 212, width: 320, height: 180, name: 'shot_04.png' },
  { index: 5, row: 1, column: 1, x: 352, y: 212, width: 320, height: 180, name: 'shot_05.png' },
  { index: 6, row: 1, column: 2, x: 684, y: 212, width: 320, height: 180, name: 'shot_06.png' }
];
const TUTORIAL_COLORS = [
  [220, 30, 30, 255],
  [30, 180, 40, 255],
  [30, 60, 220, 255],
  [230, 200, 20, 255],
  [20, 200, 210, 255],
  [240, 240, 240, 255]
];
const SMALL_FRAMES = [
  { index: 1, row: 0, column: 0, x: 1, y: 1, width: 3, height: 2, name: 'shot_01.png' },
  { index: 2, row: 0, column: 1, x: 6, y: 1, width: 3, height: 2, name: 'shot_02.png' },
  { index: 3, row: 1, column: 0, x: 1, y: 5, width: 3, height: 2, name: 'shot_03.png' },
  { index: 4, row: 1, column: 1, x: 6, y: 5, width: 3, height: 2, name: 'shot_04.png' }
];

function paintRect(rgba, imageWidth, x, y, width, height, color) {
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const offset = ((y + row) * imageWidth + x + column) * 4;
      rgba[offset] = color[0];
      rgba[offset + 1] = color[1];
      rgba[offset + 2] = color[2];
      rgba[offset + 3] = color[3];
    }
  }
}

function sliceRect(rgba, imageWidth, frame) {
  const out = new Uint8Array(frame.width * frame.height * 4);
  for (let row = 0; row < frame.height; row++) {
    const from = ((frame.y + row) * imageWidth + frame.x) * 4;
    out.set(rgba.subarray(from, from + frame.width * 4), row * frame.width * 4);
  }
  return out;
}

function paintPanels(width, height, frames, colors) {
  const rgba = new Uint8Array(width * height * 4);
  paintRect(rgba, width, 0, 0, width, height, MARKER);
  frames.forEach((frame, index) => {
    paintRect(rgba, width, frame.x, frame.y, frame.width, frame.height, colors[index]);
    paintRect(rgba, width, frame.x, frame.y, 1, 1, [frame.index, 20, 30, 255]);
    paintRect(rgba, width, frame.x + 1, frame.y, 1, 1, [frame.index, 40, 50, 255]);
  });
  if (frames[0].width > 10 && frames[0].height > 10) {
    paintRect(rgba, width, frames[0].x + 10, frames[0].y + 10, 1, 1, [9, 8, 7, 128]);
  }
  const crops = {};
  for (const frame of frames) crops[frame.index] = sliceRect(rgba, width, frame);
  return { rgba, crops };
}

function chunk(type, data) {
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const checksum = Buffer.alloc(4);
  checksum.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([length, body, checksum]);
}

function encodePng(rgba, width, height) {
  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * 4 + 1)] = 0;
    Buffer.from(rgba.buffer, rgba.byteOffset + y * width * 4, width * 4).copy(raw, y * (width * 4 + 1) + 1);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0))
  ]);
}

function decodePng(png) {
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (!png.subarray(0, 8).equals(signature)) throw new Error('Not a PNG');
  let offset = 8;
  let width = 0;
  let height = 0;
  const idat = [];
  while (offset < png.length) {
    const length = png.readUInt32BE(offset);
    const type = png.subarray(offset + 4, offset + 8).toString('ascii');
    const data = png.subarray(offset + 8, offset + 8 + length);
    const expected = crc32(png.subarray(offset + 4, offset + 8 + length));
    if (expected !== png.readUInt32BE(offset + 8 + length)) throw new Error('Bad PNG checksum for ' + type);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      if (data[8] !== 8 || data[9] !== 6 || data[10] !== 0 || data[11] !== 0 || data[12] !== 0) {
        throw new Error('Only 8-bit non-interlaced RGBA PNG files written by this fixture can be decoded here.');
      }
    } else if (type === 'IDAT') {
      idat.push(data);
    } else if (type === 'IEND') {
      break;
    }
    offset += 12 + length;
  }
  const inflated = zlib.inflateSync(Buffer.concat(idat));
  const rgba = new Uint8Array(width * height * 4);
  const stride = width * 4 + 1;
  if (inflated.length !== stride * height) throw new Error('Unexpected PNG scanlines');
  for (let y = 0; y < height; y++) {
    if (inflated[y * stride] !== 0) throw new Error('PNG filter is not None');
    inflated.copy(rgba, y * width * 4, y * stride + 1, y * stride + 1 + width * 4);
  }
  return { width, height, rgba };
}

function tutorialGrid() {
  const width = 1024;
  const height = 412;
  const painted = paintPanels(width, height, TUTORIAL_FRAMES, TUTORIAL_COLORS);
  return {
    width,
    height,
    settings: { rows: 2, columns: 3, padding: 20, gapX: 12, gapY: 12 },
    frames: TUTORIAL_FRAMES,
    colors: TUTORIAL_COLORS,
    png: encodePng(painted.rgba, width, height),
    ...painted
  };
}

function smallGrid() {
  const width = 10;
  const height = 8;
  const colors = [
    [10, 0, 0, 255],
    [0, 10, 0, 255],
    [0, 0, 10, 255],
    [10, 10, 0, 255]
  ];
  const painted = paintPanels(width, height, SMALL_FRAMES, colors);
  return {
    width,
    height,
    settings: { rows: 2, columns: 2, padding: 1, gapX: 2, gapY: 2 },
    frames: SMALL_FRAMES,
    colors,
    png: encodePng(painted.rgba, width, height),
    ...painted
  };
}

module.exports = {
  MARKER,
  TUTORIAL_FRAMES,
  SMALL_FRAMES,
  encodePng,
  decodePng,
  tutorialGrid,
  smallGrid
};
