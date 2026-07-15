import * as THREE from "three";

// ---------- Easing ----------
export const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
export const easeOut = (t) => 1 - Math.pow(1 - t, 3);
export const easeOutBack = (t) => {
  const c1 = 1.70158, c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
};

// ---------- Tween manager ----------
const tweens = [];
export function tween({ dur, onUpdate, onDone, ease = easeInOut, delay = 0 }) {
  const tw = { t: -delay, dur, onUpdate, onDone, ease, alive: true };
  tweens.push(tw);
  return tw;
}
export function cancelTween(tw) { if (tw) tw.alive = false; }
export function clearTweens() { tweens.length = 0; }
export function updateTweens(dt) {
  for (let i = tweens.length - 1; i >= 0; i--) {
    const tw = tweens[i];
    if (!tw.alive) { tweens.splice(i, 1); continue; }
    tw.t += dt;
    if (tw.t < 0) continue;
    const p = tw.dur <= 0 ? 1 : Math.min(tw.t / tw.dur, 1);
    tw.onUpdate(tw.ease(p), p);
    if (p >= 1) { tw.alive = false; tweens.splice(i, 1); if (tw.onDone) tw.onDone(); }
  }
}

// ---------- Colormap (dark -> blue -> cyan -> yellow), feature-map style ----------
const CMAP = [
  [0.0, [16, 26, 60]],
  [0.25, [34, 74, 150]],
  [0.5, [34, 160, 210]],
  [0.72, [56, 220, 180]],
  [1.0, [255, 214, 78]],
];
export function colormap(v) {
  v = Math.max(0, Math.min(1, v));
  for (let i = 1; i < CMAP.length; i++) {
    if (v <= CMAP[i][0]) {
      const [a0, c0] = CMAP[i - 1], [a1, c1] = CMAP[i];
      const t = (v - a0) / (a1 - a0);
      return [
        c0[0] + (c1[0] - c0[0]) * t,
        c0[1] + (c1[1] - c0[1]) * t,
        c0[2] + (c1[2] - c0[2]) * t,
      ];
    }
  }
  return CMAP[CMAP.length - 1][1];
}

// ---------- Build an RGBA DataTexture from a channel of a tensor ----------
// data: Float32Array, offset+ (w*h) values. mode: 'gray' | 'cmap'. signed: map [-max,max].
export function makeTexture(data, offset, w, h, { mode = "cmap", signed = false, reveal = 1 } = {}) {
  const n = w * h;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) { const v = data[offset + i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  if (signed) { const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-6); lo = -m; hi = m; }
  const range = hi - lo || 1e-6;
  const buf = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const norm = (data[offset + i] - lo) / range;
    let r, g, b;
    if (mode === "gray") { r = g = b = norm * 255; }
    else { const c = colormap(Math.pow(norm, 0.72)); r = c[0]; g = c[1]; b = c[2]; } // gamma lifts faint deep features
    buf[i * 4] = r; buf[i * 4 + 1] = g; buf[i * 4 + 2] = b; buf[i * 4 + 3] = 255;
  }
  const tex = new THREE.DataTexture(buf, w, h, THREE.RGBAFormat);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.needsUpdate = true;
  tex.flipY = true;
  tex._buf = buf; tex._w = w; tex._h = h; // for reveal updates
  return tex;
}

// Set per-pixel alpha for a "reveal" wipe based on raster progress [0..1]
export function revealTexture(tex, progress) {
  const n = tex._w * tex._h;
  const cut = Math.floor(progress * n);
  for (let i = 0; i < n; i++) tex._buf[i * 4 + 3] = i < cut ? 255 : 0;
  tex.needsUpdate = true;
}

// ---------- Grid lines overlay (graph-paper look) ----------
export function makeGridLines(w, h, cell, color = 0x2a3a66, opacity = 0.5) {
  const pts = [];
  const W = w * cell, H = h * cell;
  const x0 = -W / 2, y0 = -H / 2;
  for (let x = 0; x <= w; x++) { pts.push(x0 + x * cell, y0, 0, x0 + x * cell, y0 + H, 0); }
  for (let y = 0; y <= h; y++) { pts.push(x0, y0 + y * cell, 0, x0 + W, y0 + y * cell, 0); }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  return new THREE.LineSegments(geo, mat);
}

// Fade a group's materials via .opacity
export function setGroupOpacity(obj, o) {
  obj.traverse((c) => {
    if (c.material) {
      const mats = Array.isArray(c.material) ? c.material : [c.material];
      mats.forEach((m) => { m.transparent = true; m.opacity = o; });
    }
  });
}

export function lerp(a, b, t) { return a + (b - a) * t; }
