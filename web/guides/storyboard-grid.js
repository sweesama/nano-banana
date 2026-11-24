(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.StoryboardGrid = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  var MAX_PANELS = 100;
  var MAX_EDGE = 8192;
  var MAX_PIXELS = 32000000;
  var ZIP_DOS_TIME = 0;
  var ZIP_DOS_DATE = ((2026 - 1980) << 9) | (11 << 5) | 24;
  var FIELDS = [
    ['rows', 'Rows', 1],
    ['columns', 'Columns', 1],
    ['padding', 'Outer padding', 0],
    ['gapX', 'Horizontal gap', 0],
    ['gapY', 'Vertical gap', 0]
  ];

  function wholeNumberMessage(label) {
    return label + ' must be a whole number, with no decimals, signs, or spaces.';
  }

  function minimumMessage(label, minimum) {
    return label + ' must be ' + (minimum === 0 ? 'zero or greater' : 'at least 1') + '.';
  }

  function fail(errors) {
    return { ok: false, errors: errors, notes: [], frames: [] };
  }

  function readCount(value, label, minimum, errors) {
    if (typeof value !== 'number' || !Number.isSafeInteger(value)) {
      errors.push(wholeNumberMessage(label));
      return 0;
    }
    if (value < minimum) {
      errors.push(minimumMessage(label, minimum));
      return 0;
    }
    return value;
  }

  function readPixelSize(value, label, errors) {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) {
      errors.push(label + ' must be a positive whole number of pixels.');
      return 0;
    }
    return value;
  }

  function parseSettings(raw) {
    var errors = [];
    var settings = {};
    for (var i = 0; i < FIELDS.length; i++) {
      var key = FIELDS[i][0];
      var label = FIELDS[i][1];
      var minimum = FIELDS[i][2];
      var text = raw && Object.prototype.hasOwnProperty.call(raw, key) ? raw[key] : '';
      if (typeof text !== 'string' || !/^(?:0|[1-9][0-9]*)$/.test(text)) {
        errors.push(wholeNumberMessage(label));
        continue;
      }
      var number = Number(text);
      if (!Number.isSafeInteger(number)) {
        errors.push(label + ' is too large.');
        continue;
      }
      if (number < minimum) {
        errors.push(minimumMessage(label, minimum));
        continue;
      }
      settings[key] = number;
    }
    if (errors.length) return { ok: false, errors: errors };
    return { ok: true, settings: settings };
  }

  function imageSizeError(width, height) {
    if (!Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1) {
      return 'Image width and height must be positive whole numbers of pixels.';
    }
    if (width > MAX_EDGE || height > MAX_EDGE || width * height > MAX_PIXELS) {
      return 'This image is ' + width + '×' + height + ' px. The page stops above ' + MAX_EDGE + ' px on a side or ' + MAX_PIXELS + ' pixels.';
    }
    return '';
  }

  function planGrid(input) {
    var source = input || {};
    var errors = [];
    var width = readPixelSize(source.width, 'Image width', errors);
    var height = readPixelSize(source.height, 'Image height', errors);
    var rows = readCount(source.rows, 'Rows', 1, errors);
    var columns = readCount(source.columns, 'Columns', 1, errors);
    var padding = readCount(source.padding, 'Outer padding', 0, errors);
    var gapX = readCount(source.gapX, 'Horizontal gap', 0, errors);
    var gapY = readCount(source.gapY, 'Vertical gap', 0, errors);
    if (errors.length) return fail(errors);
    var sizeError = imageSizeError(width, height);
    if (sizeError) return fail([sizeError]);
    if (rows > MAX_PANELS || columns > MAX_PANELS || rows * columns > MAX_PANELS) {
      var count = rows > MAX_PANELS || columns > MAX_PANELS ? 0 : rows * columns;
      errors.push(count
        ? count + ' panels is above the ' + MAX_PANELS + '-panel limit.'
        : 'Rows or columns are above the ' + MAX_PANELS + '-panel limit.');
      return fail(errors);
    }

    var horizontalGaps = columns - 1;
    var verticalGaps = rows - 1;
    var remainingWidth = width - 2 * padding - horizontalGaps * gapX;
    var remainingHeight = height - 2 * padding - verticalGaps * gapY;
    if (remainingWidth <= 0) {
      errors.push(remainingMessage('width', remainingWidth, padding, horizontalGaps, gapX));
    } else if (remainingWidth % columns !== 0) {
      errors.push(leftoverMessage('width', remainingWidth, columns));
    }
    if (remainingHeight <= 0) {
      errors.push(remainingMessage('height', remainingHeight, padding, verticalGaps, gapY));
    } else if (remainingHeight % rows !== 0) {
      errors.push(leftoverMessage('height', remainingHeight, rows));
    }
    if (errors.length) return fail(errors);

    var cellWidth = remainingWidth / columns;
    var cellHeight = remainingHeight / rows;
    var frames = [];
    for (var row = 0; row < rows; row++) {
      for (var column = 0; column < columns; column++) {
        var index = row * columns + column + 1;
        var frame = {
          index: index,
          row: row,
          column: column,
          x: padding + column * (cellWidth + gapX),
          y: padding + row * (cellHeight + gapY),
          width: cellWidth,
          height: cellHeight,
          name: 'shot_' + String(index).padStart(2, '0') + '.png'
        };
        if (frame.x < 0 || frame.y < 0 || frame.x + frame.width > width || frame.y + frame.height > height) {
          return fail(['Panel ' + index + ' extends outside the image.']);
        }
        frames.push(frame);
      }
    }
    var notes = [];
    if (columns === 1 && gapX > 0) notes.push('Horizontal gap is unused because there is one column.');
    if (rows === 1 && gapY > 0) notes.push('Vertical gap is unused because there is one row.');
    return {
      ok: true,
      errors: [],
      notes: notes,
      width: width,
      height: height,
      rows: rows,
      columns: columns,
      padding: padding,
      gapX: gapX,
      gapY: gapY,
      remainingWidth: remainingWidth,
      remainingHeight: remainingHeight,
      cellWidth: cellWidth,
      cellHeight: cellHeight,
      frames: frames
    };
  }

  function remainingMessage(axis, remaining, padding, gapCount, gap) {
    var text = 'After ' + padding + ' px of outer padding on each side';
    if (gapCount > 0) {
      var name = axis === 'width' ? 'horizontal' : 'vertical';
      text += ' and ' + gapCount + ' ' + name + ' gap' + (gapCount === 1 ? '' : 's') + ' of ' + gap + ' px';
    }
    return text + ', the remaining ' + axis + ' is ' + remaining + ' px.';
  }

  function leftoverMessage(axis, remaining, parts) {
    var leftover = remaining % parts;
    var noun = axis === 'width' ? 'column' : 'row';
    return 'The remaining ' + axis + ' is ' + remaining + ' px. It does not divide evenly into ' + parts + ' ' + noun + (parts === 1 ? '' : 's') + ' (' + leftover + ' px left over). This tool does not round the cell, because that would pull in a strip of the next panel or the gap.';
  }

  function cropRgba(image, rect) {
    if (!image || !image.data || image.data.length !== image.width * image.height * 4) {
      return { ok: false, error: 'The decoded image is incomplete.' };
    }
    var x = rect.x;
    var y = rect.y;
    var width = rect.width;
    var height = rect.height;
    var label = rect.index ? 'Panel ' + rect.index : 'A panel';
    if (![x, y, width, height].every(function (value) { return Number.isSafeInteger(value) && value >= 0; }) || width < 1 || height < 1 || x + width > image.width || y + height > image.height) {
      return { ok: false, error: label + ' extends outside the image.' };
    }
    var out = new Uint8Array(width * height * 4);
    for (var row = 0; row < height; row++) {
      var from = ((y + row) * image.width + x) * 4;
      out.set(image.data.subarray(from, from + width * 4), row * width * 4);
    }
    return { ok: true, bytes: out };
  }

  function buildCrops(image, plan) {
    if (!plan || plan.ok !== true || !plan.frames || !plan.frames.length) {
      return { ok: false, error: 'The grid is not valid.' };
    }
    if (!image || image.width !== plan.width || image.height !== plan.height) {
      return { ok: false, error: 'The decoded image size does not match the grid.' };
    }
    var crops = {};
    for (var i = 0; i < plan.frames.length; i++) {
      var cropped = cropRgba(image, plan.frames[i]);
      if (!cropped.ok) return { ok: false, error: cropped.error };
      crops[plan.frames[i].index] = cropped.bytes;
    }
    return { ok: true, crops: crops };
  }

  function copyBytes(bytes) {
    return new Uint8Array(bytes);
  }

  function createSession() {
    var generation = 0;
    var revision = 0;
    var current = null;

    function dropGeneration() {
      generation += 1;
      revision += 1;
      current = null;
      return { generation: generation, revision: revision };
    }

    return {
      generation: function () { return generation; },
      revision: function () { return revision; },
      replaceImage: dropGeneration,
      clear: dropGeneration,
      invalidate: function (generationId) {
        if (generationId !== generation) {
          return { ok: false, error: 'This image is no longer current. Nothing was exported.' };
        }
        revision += 1;
        current = null;
        return { ok: true, generation: generation, revision: revision };
      },
      commit: function (generationId, plan, crops) {
        if (generationId !== generation) {
          return { ok: false, error: 'This image is no longer current. Nothing was exported.' };
        }
        revision += 1;
        current = null;
        if (!plan || plan.ok !== true || !plan.frames || !plan.frames.length) {
          return { ok: false, error: 'The grid is not valid. Nothing was exported.' };
        }
        var stored = {};
        for (var i = 0; i < plan.frames.length; i++) {
          var frame = plan.frames[i];
          var bytes = crops && crops[frame.index];
          if (!bytes || bytes.length !== frame.width * frame.height * 4) {
            return { ok: false, error: 'Panel ' + frame.index + ' did not produce ' + frame.width + '×' + frame.height + ' pixels. Nothing was exported.' };
          }
          stored[frame.index] = copyBytes(bytes);
        }
        current = { generation: generationId, revision: revision, plan: plan, crops: stored };
        return { ok: true, generation: generation, revision: revision };
      },
      exportSelection: function (generationId, revisionId, indexes) {
        if (!current || current.generation !== generationId || generationId !== generation || current.revision !== revisionId) {
          return { ok: false, error: 'No current crop is ready. Choose an image and a valid grid before downloading.' };
        }
        if (!Array.isArray(indexes) || !indexes.length) {
          return { ok: false, error: 'Select at least one panel. Nothing was downloaded.' };
        }
        var wanted = new Set();
        for (var i = 0; i < indexes.length; i++) {
          if (!Number.isSafeInteger(indexes[i]) || !current.crops[indexes[i]]) {
            return { ok: false, error: 'A selected panel is not in the current grid. Nothing was downloaded.' };
          }
          wanted.add(indexes[i]);
        }
        var frames = current.plan.frames.filter(function (frame) { return wanted.has(frame.index); }).map(function (frame) {
          return {
            index: frame.index,
            name: frame.name,
            width: frame.width,
            height: frame.height,
            bytes: copyBytes(current.crops[frame.index])
          };
        });
        return {
          ok: true,
          kind: frames.length === 1 ? 'png' : 'zip',
          name: frames.length === 1 ? frames[0].name : 'storyboard-shots.zip',
          frames: frames
        };
      }
    };
  }

  var crcTable = null;
  function crc32(bytes) {
    if (!crcTable) {
      crcTable = new Uint32Array(256);
      for (var n = 0; n < 256; n++) {
        var c = n;
        for (var k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
        crcTable[n] = c >>> 0;
      }
    }
    var crc = 0xFFFFFFFF;
    for (var i = 0; i < bytes.length; i++) crc = crcTable[(crc ^ bytes[i]) & 0xFF] ^ (crc >>> 8);
    return (crc ^ 0xFFFFFFFF) >>> 0;
  }

  function alloc(size) {
    var buffer = new ArrayBuffer(size);
    return { bytes: new Uint8Array(buffer), view: new DataView(buffer) };
  }

  function concat(parts) {
    var total = 0;
    for (var i = 0; i < parts.length; i++) total += parts[i].length;
    var out = new Uint8Array(total);
    var offset = 0;
    for (var j = 0; j < parts.length; j++) {
      out.set(parts[j], offset);
      offset += parts[j].length;
    }
    return out;
  }

  function zipStore(files) {
    if (!Array.isArray(files) || files.length < 1 || files.length > MAX_PANELS) {
      throw new Error('ZIP needs between 1 and ' + MAX_PANELS + ' files.');
    }
    var encoder = new TextEncoder();
    var seen = new Set();
    var prepared = files.map(function (file) {
      if (!file || typeof file.name !== 'string' || !/^[A-Za-z0-9._-]{1,64}$/.test(file.name)) {
        throw new Error('ZIP entry name is not a plain file name.');
      }
      if (seen.has(file.name)) throw new Error('Duplicate ZIP entry: ' + file.name);
      seen.add(file.name);
      if (!(file.bytes instanceof Uint8Array)) throw new Error('ZIP entry is missing bytes.');
      if (file.bytes.length > 64 * 1024 * 1024) throw new Error('ZIP entry is too large.');
      return { name: file.name, nameBytes: encoder.encode(file.name), bytes: file.bytes, crc: crc32(file.bytes) };
    });
    var parts = [];
    var centralParts = [];
    var offset = 0;
    for (var i = 0; i < prepared.length; i++) {
      var item = prepared[i];
      var local = alloc(30 + item.nameBytes.length);
      local.view.setUint32(0, 0x04034b50, true);
      local.view.setUint16(4, 20, true);
      local.view.setUint16(6, 0x0800, true);
      local.view.setUint16(8, 0, true);
      local.view.setUint16(10, ZIP_DOS_TIME, true);
      local.view.setUint16(12, ZIP_DOS_DATE, true);
      local.view.setUint32(14, item.crc, true);
      local.view.setUint32(18, item.bytes.length, true);
      local.view.setUint32(22, item.bytes.length, true);
      local.view.setUint16(26, item.nameBytes.length, true);
      local.view.setUint16(28, 0, true);
      local.bytes.set(item.nameBytes, 30);
      parts.push(local.bytes, item.bytes);
      var central = alloc(46 + item.nameBytes.length);
      central.view.setUint32(0, 0x02014b50, true);
      central.view.setUint16(4, 20, true);
      central.view.setUint16(6, 20, true);
      central.view.setUint16(8, 0x0800, true);
      central.view.setUint16(10, 0, true);
      central.view.setUint16(12, ZIP_DOS_TIME, true);
      central.view.setUint16(14, ZIP_DOS_DATE, true);
      central.view.setUint32(16, item.crc, true);
      central.view.setUint32(20, item.bytes.length, true);
      central.view.setUint32(24, item.bytes.length, true);
      central.view.setUint16(28, item.nameBytes.length, true);
      central.view.setUint16(30, 0, true);
      central.view.setUint16(32, 0, true);
      central.view.setUint16(34, 0, true);
      central.view.setUint16(36, 0, true);
      central.view.setUint32(38, 0, true);
      central.view.setUint32(42, offset, true);
      central.bytes.set(item.nameBytes, 46);
      centralParts.push(central.bytes);
      offset += local.bytes.length + item.bytes.length;
    }
    var directory = concat(centralParts);
    var eocd = alloc(22);
    eocd.view.setUint32(0, 0x06054b50, true);
    eocd.view.setUint16(8, prepared.length, true);
    eocd.view.setUint16(10, prepared.length, true);
    eocd.view.setUint32(12, directory.length, true);
    eocd.view.setUint32(16, offset, true);
    return concat(parts.concat([directory, eocd.bytes]));
  }

  function readZip(bytes) {
    var data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    if (data.length < 22) throw new Error('ZIP is too small.');
    var view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    var eocd = data.length - 22;
    if (view.getUint32(eocd, true) !== 0x06054b50) throw new Error('ZIP end record was not found.');
    if (view.getUint16(eocd + 4, true) !== 0 || view.getUint16(eocd + 6, true) !== 0) throw new Error('ZIP spans more than one disk.');
    var entries = view.getUint16(eocd + 8, true);
    if (entries < 1 || entries !== view.getUint16(eocd + 10, true)) throw new Error('ZIP entry count is unexpected.');
    var directorySize = view.getUint32(eocd + 12, true);
    var directoryOffset = view.getUint32(eocd + 16, true);
    if (view.getUint16(eocd + 20, true) !== 0) throw new Error('ZIP comment was not expected.');
    if (directoryOffset + directorySize !== eocd) throw new Error('ZIP directory does not meet the end record.');
    var files = [];
    var cursor = directoryOffset;
    for (var i = 0; i < entries; i++) {
      if (cursor + 46 > eocd || view.getUint32(cursor, true) !== 0x02014b50) throw new Error('ZIP directory entry is invalid.');
      if ((view.getUint16(cursor + 8, true) & 0x0008) !== 0) throw new Error('ZIP data descriptor was not expected.');
      var crc = view.getUint32(cursor + 16, true);
      var compressed = view.getUint32(cursor + 20, true);
      var uncompressed = view.getUint32(cursor + 24, true);
      var nameLength = view.getUint16(cursor + 28, true);
      if (view.getUint16(cursor + 10, true) !== 0 || compressed !== uncompressed) throw new Error('ZIP entry is not stored uncompressed.');
      if (view.getUint16(cursor + 30, true) !== 0 || view.getUint16(cursor + 32, true) !== 0) throw new Error('ZIP extra fields were not expected.');
      var nameStart = cursor + 46;
      var nameEnd = nameStart + nameLength;
      if (nameEnd > eocd) throw new Error('ZIP directory name is truncated.');
      var name = new TextDecoder().decode(data.subarray(nameStart, nameEnd));
      var localOffset = view.getUint32(cursor + 42, true);
      if (localOffset + 30 > directoryOffset || view.getUint32(localOffset, true) !== 0x04034b50) throw new Error('ZIP local header is invalid.');
      if (view.getUint16(localOffset + 8, true) !== 0 || view.getUint32(localOffset + 14, true) !== crc || view.getUint32(localOffset + 18, true) !== uncompressed || view.getUint32(localOffset + 22, true) !== uncompressed) {
        throw new Error('ZIP local header does not match the directory.');
      }
      if (view.getUint16(localOffset + 26, true) !== nameLength || view.getUint16(localOffset + 28, true) !== 0) throw new Error('ZIP local name does not match the directory.');
      var localName = new TextDecoder().decode(data.subarray(localOffset + 30, localOffset + 30 + nameLength));
      if (localName !== name) throw new Error('ZIP local name does not match the directory.');
      var payloadStart = localOffset + 30 + nameLength;
      var payloadEnd = payloadStart + uncompressed;
      if (payloadEnd > directoryOffset) throw new Error('ZIP payload overflows.');
      var payload = data.subarray(payloadStart, payloadEnd);
      if (crc32(payload) !== crc) throw new Error('ZIP checksum does not match ' + name + '.');
      files.push({ name: name, bytes: new Uint8Array(payload) });
      cursor = nameEnd;
    }
    if (cursor !== eocd) throw new Error('ZIP directory length does not match.');
    return files;
  }

  return {
    MAX_PANELS: MAX_PANELS,
    MAX_EDGE: MAX_EDGE,
    MAX_PIXELS: MAX_PIXELS,
    ZIP_DOS_TIME: ZIP_DOS_TIME,
    ZIP_DOS_DATE: ZIP_DOS_DATE,
    parseSettings: parseSettings,
    imageSizeError: imageSizeError,
    planGrid: planGrid,
    cropRgba: cropRgba,
    buildCrops: buildCrops,
    createSession: createSession,
    crc32: crc32,
    zipStore: zipStore,
    readZip: readZip
  };
});
