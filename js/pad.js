// Drawing pad: neon-blue stroke shown to the user, while a hidden "clean"
// canvas (plain white on black, no glow) is used for accurate MNIST extraction.

export function initPad() {
  const canvas = document.getElementById("pad");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  // hidden clean canvas, same size, no glow -> reliable pixel intensities
  const clean = document.createElement("canvas");
  clean.width = canvas.width; clean.height = canvas.height;
  const cctx = clean.getContext("2d", { willReadFrequently: true });

  let drawing = false, last = null, dirty = false;
  const LW = 24;

  function reset() {
    ctx.save(); ctx.shadowBlur = 0; ctx.fillStyle = "#0a0f22";
    ctx.fillRect(0, 0, canvas.width, canvas.height); ctx.restore();
    cctx.fillStyle = "#000"; cctx.fillRect(0, 0, clean.width, clean.height);
    dirty = false;
  }
  function styleStroke() {
    for (const c of [ctx, cctx]) { c.lineWidth = LW; c.lineCap = "round"; c.lineJoin = "round"; }
    // visible: neon blue with glow
    ctx.strokeStyle = "#d8fbff"; ctx.shadowColor = "#38e8ff"; ctx.shadowBlur = 18;
    ctx.fillStyle = "#d8fbff";
    // clean: pure white, no glow
    cctx.strokeStyle = "#fff"; cctx.shadowBlur = 0; cctx.fillStyle = "#fff";
  }
  reset(); styleStroke();

  function pos(e) {
    const r = canvas.getBoundingClientRect();
    const t = e.touches ? e.touches[0] : e;
    return {
      x: ((t.clientX - r.left) / r.width) * canvas.width,
      y: ((t.clientY - r.top) / r.height) * canvas.height,
    };
  }
  function stroke(a, b) {
    for (const c of [ctx, cctx]) {
      c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
    }
  }
  function dot(p) {
    for (const c of [ctx, cctx]) { c.beginPath(); c.arc(p.x, p.y, LW / 2, 0, 7); c.fill(); }
  }
  function start(e) { e.preventDefault(); drawing = true; last = pos(e); dot(last); dirty = true; }
  function move(e) { if (!drawing) return; e.preventDefault(); const p = pos(e); stroke(last, p); last = p; dirty = true; }
  function end() { drawing = false; }

  canvas.addEventListener("mousedown", start);
  canvas.addEventListener("mousemove", move);
  window.addEventListener("mouseup", end);
  canvas.addEventListener("touchstart", start, { passive: false });
  canvas.addEventListener("touchmove", move, { passive: false });
  canvas.addEventListener("touchend", end);

  // MNIST-style 28x28 Float32Array in [0,1] from the clean canvas.
  function extract() {
    const src = cctx.getImageData(0, 0, clean.width, clean.height).data;
    const W = clean.width, H = clean.height;
    let minX = W, minY = H, maxX = 0, maxY = 0, any = false;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        if (src[(y * W + x) * 4] > 50) {
          any = true;
          if (x < minX) minX = x; if (x > maxX) maxX = x;
          if (y < minY) minY = y; if (y > maxY) maxY = y;
        }
      }
    if (!any) return null;

    const bw = maxX - minX + 1, bh = maxY - minY + 1;
    const scale = 20 / Math.max(bw, bh);
    const tmp = document.createElement("canvas");
    tmp.width = 28; tmp.height = 28;
    const tctx = tmp.getContext("2d", { willReadFrequently: true });
    tctx.fillStyle = "#000"; tctx.fillRect(0, 0, 28, 28);
    tctx.imageSmoothingEnabled = true;
    const dw = bw * scale, dh = bh * scale;
    tctx.drawImage(clean, minX, minY, bw, bh, (28 - dw) / 2, (28 - dh) / 2, dw, dh);

    const d = tctx.getImageData(0, 0, 28, 28).data;
    const arr = new Float32Array(784);
    let sx = 0, sy = 0, sm = 0;
    for (let i = 0; i < 784; i++) {
      const v = d[i * 4] / 255;
      arr[i] = v;
      const x = i % 28, y = (i / 28) | 0;
      sx += x * v; sy += y * v; sm += v;
    }
    if (sm > 0) {
      const cx = sx / sm, cy = sy / sm;
      const shiftX = Math.round(14 - cx), shiftY = Math.round(14 - cy);
      if (shiftX || shiftY) {
        const shifted = new Float32Array(784);
        for (let y = 0; y < 28; y++)
          for (let x = 0; x < 28; x++) {
            const nx = x + shiftX, ny = y + shiftY;
            if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28)
              shifted[ny * 28 + nx] = arr[y * 28 + x];
          }
        return shifted;
      }
    }
    return arr;
  }

  function clear() { reset(); styleStroke(); }
  function hasInk() { return dirty; }
  return { extract, clear, hasInk };
}
