import * as THREE from "three";
import { makeTexture, makeGridLines, colormap, setGroupOpacity } from "./util.js";

// Each stage sits at its own X (side by side, left -> right) and steps slightly
// back in Z for depth. Laying stages along X (instead of stacking on one axis)
// stops earlier layers from occluding the current one.
export const POS = {
  input:   { x: 0,   z: 0 },
  conv1:   { x: 26,  z: -5 },
  pool1:   { x: 50,  z: -9 },
  conv2:   { x: 80,  z: -13 },
  pool2:   { x: 112, z: -17 },
  flatten: { x: 140, z: -21 },
  fc:      { x: 168, z: -25 },
};

const CI = 0.5; // input cell size

// ---- Input as a 28x28 heightfield of cubes (hero look) ----
export function buildInput(input, mat3d) {
  const group = new THREE.Group();
  const w = 28, h = 28, n = w * h;
  const geo = new THREE.BoxGeometry(CI * 0.86, CI * 0.86, 1);
  const mesh = new THREE.InstancedMesh(geo, mat3d, n);
  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  const heights = new Float32Array(n);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const v = input.data[i];
      heights[i] = 0.06 + v * 2.6;
      dummy.position.set(-w / 2 * CI + (x + 0.5) * CI, h / 2 * CI - (y + 0.5) * CI, heights[i] / 2);
      dummy.scale.set(1, 1, heights[i]);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      const g = 0.12 + v * 0.88;
      color.setRGB(g * 0.7 + v * 0.3, g, g * 0.9 + 0.1);
      mesh.setColorAt(i, color);
    }
  mesh.instanceColor.needsUpdate = true;
  mesh.userData = { heights, w, h, dummy };
  group.add(mesh);
  const grid = makeGridLines(w, h, CI, 0x2a3a66, 0.35);
  grid.position.z = 0.02;
  group.add(grid);
  group.position.set(POS.input.x, 0, POS.input.z);
  return { group, mesh };
}

// ---- RGB channel split planes (educational) ----
export function buildRGB(input) {
  const group = new THREE.Group();
  const tex = makeTexture(input.data, 0, 28, 28, { mode: "gray" });
  const size = 28 * CI;
  const tints = [0xff3355, 0x33ff77, 0x3388ff];
  const planes = tints.map((c) => {
    const m = new THREE.MeshBasicMaterial({
      map: tex.clone(), color: c, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    m.map.needsUpdate = true;
    const p = new THREE.Mesh(new THREE.PlaneGeometry(size, size), m);
    group.add(p);
    return p;
  });
  group.position.set(POS.input.x, 0, POS.input.z);
  return { group, planes };
}

// ---- 3x3 kernel (conv1 out-channel 0 weights) as colored cubes ----
export function buildKernel(kernel) {
  const group = new THREE.Group();
  const cubes = [];
  const cs = CI * 0.9;
  for (let ky = 0; ky < 3; ky++)
    for (let kx = 0; kx < 3; kx++) {
      const w = kernel[ky * 3 + kx];
      const c = colormap((w / 0.6) * 0.5 + 0.5);
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255),
        emissive: new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255),
        emissiveIntensity: 0.4, transparent: true, opacity: 0, metalness: 0.3, roughness: 0.4,
      });
      const cube = new THREE.Mesh(new THREE.BoxGeometry(cs, cs, cs), mat);
      cube.position.set((kx - 1) * CI, -(ky - 1) * CI, 0);
      group.add(cube);
      cubes.push(cube);
    }
  // receptive-field highlight on the input
  const hlMat = new THREE.MeshBasicMaterial({ color: 0x38e8ff, transparent: true, opacity: 0, side: THREE.DoubleSide });
  const hl = new THREE.Mesh(new THREE.PlaneGeometry(CI * 3, CI * 3), hlMat);
  const frame = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.PlaneGeometry(CI * 3, CI * 3)),
    new THREE.LineBasicMaterial({ color: 0x38e8ff, transparent: true, opacity: 0 })
  );
  group.position.set(POS.input.x, 0, POS.input.z + 3.2);
  return { group, cubes, hl, hlMat, frame };
}

// ---- A layer of feature-map planes arranged in a grid ----
export function buildFeatureLayer(data, count, w, h, { cols, cell, gap, pos, signed = false }) {
  const group = new THREE.Group();
  const mapW = w * cell, mapH = h * cell;
  const rows = Math.ceil(count / cols);
  const totW = cols * mapW + (cols - 1) * gap;
  const totH = rows * mapH + (rows - 1) * gap;
  const planes = [];
  for (let i = 0; i < count; i++) {
    const cx = i % cols, cy = (i / cols) | 0;
    const px = -totW / 2 + cx * (mapW + gap) + mapW / 2;
    const py = totH / 2 - cy * (mapH + gap) - mapH / 2;
    const tex = makeTexture(data, i * w * h, w, h, { mode: "cmap", signed });
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0, side: THREE.DoubleSide });
    const sub = new THREE.Group();
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(mapW, mapH), mat);
    sub.add(plane);
    // thin back slab for 3D depth — shown faintly from the start as a scaffold
    const slab = new THREE.Mesh(
      new THREE.BoxGeometry(mapW * 1.04, mapH * 1.04, 0.25),
      new THREE.MeshStandardMaterial({ color: 0x101a3a, transparent: true, opacity: 0.14, metalness: 0.2, roughness: 0.7 })
    );
    slab.position.z = -0.2;
    sub.add(slab);
    if (w <= 16) { const gl = makeGridLines(w, h, cell, 0x33507f, 0.22); gl.position.z = 0.01; sub.add(gl); }
    sub.position.set(px, py, 0);
    group.add(sub);
    planes.push({ sub, plane, slab, tex, mat });
  }
  group.position.set(pos.x, 0, pos.z);
  group.userData = { totW, totH };
  return { group, planes };
}

// ---- Flatten: 784 values as a wide strip of instanced cubes ----
export function buildFlatten(flat) {
  const group = new THREE.Group();
  const n = flat.length; // 784
  const cols = 56, rows = Math.ceil(n / cols); // 56 x 14
  const cell = 0.34;
  const geo = new THREE.BoxGeometry(cell * 0.82, cell * 0.82, cell * 0.82);
  const mat = new THREE.MeshStandardMaterial({ transparent: true, opacity: 0, metalness: 0.3, roughness: 0.5 });
  const mesh = new THREE.InstancedMesh(geo, mat, n);
  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) { if (flat[i] < lo) lo = flat[i]; if (flat[i] > hi) hi = flat[i]; }
  const range = hi - lo || 1e-6;
  const positions = [];
  for (let i = 0; i < n; i++) {
    const cx = i % cols, cy = (i / cols) | 0;
    const px = -cols / 2 * cell + (cx + 0.5) * cell;
    const py = rows / 2 * cell - (cy + 0.5) * cell;
    positions.push([px, py]);
    dummy.position.set(px, py, 0); dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    const c = colormap((flat[i] - lo) / range);
    color.setRGB(c[0] / 255, c[1] / 255, c[2] / 255);
    mesh.setColorAt(i, color);
  }
  mesh.instanceColor.needsUpdate = true;
  mesh.userData = { positions, cols, rows, cell };
  group.add(mesh);
  group.position.set(POS.flatten.x, 0, POS.flatten.z);
  return { group, mesh, positions };
}

// ---- Fully-connected: 64 hidden neurons -> 10 outputs, with connection lines ----
export function buildFC(fc1, logits, probs) {
  const group = new THREE.Group();
  const hidden = [];
  const hCols = 8, hCell = 1.3;
  const sphereGeo = new THREE.SphereGeometry(0.32, 16, 16);
  let hMax = Math.max(...fc1, 1e-6);
  for (let i = 0; i < 64; i++) {
    const cx = i % hCols, cy = (i / hCols) | 0;
    const px = -hCols / 2 * hCell + (cx + 0.5) * hCell;
    const py = 4 * hCell - cy * hCell - hCell;
    const a = fc1[i] / hMax;
    const c = colormap(a);
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255),
      emissive: new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255),
      emissiveIntensity: 0.2 + a * 0.9, transparent: true, opacity: 0, metalness: 0.4, roughness: 0.3,
    });
    const s = new THREE.Mesh(sphereGeo, mat);
    s.position.set(px, py, 0);
    group.add(s);
    hidden.push({ mesh: s, pos: s.position.clone(), act: a });
  }
  const outputs = [];
  const oGeo = new THREE.SphereGeometry(0.5, 20, 20);
  for (let d = 0; d < 10; d++) {
    const py = 5 - d * 1.05;
    const a = probs[d];
    const mat = new THREE.MeshStandardMaterial({
      color: 0x38e8ff, emissive: 0x38e8ff, emissiveIntensity: 0.15 + a * 1.4,
      transparent: true, opacity: 0, metalness: 0.4, roughness: 0.25,
    });
    const s = new THREE.Mesh(oGeo, mat);
    s.position.set(10, py, 0);
    group.add(s);
    outputs.push({ mesh: s, pos: s.position.clone(), prob: a, digit: d });
  }
  // connection lines hidden -> outputs (full 64x10 = 640)
  const linePts = [];
  const lineCols = [];
  for (const hn of hidden)
    for (const on of outputs) {
      linePts.push(hn.pos.x, hn.pos.y, hn.pos.z, on.pos.x, on.pos.y, on.pos.z);
      const b = 0.15 + hn.act * on.prob * 2;
      lineCols.push(0.2, 0.7 * b + 0.2, 0.9, 0.2, 0.7 * b + 0.2, 0.9);
    }
  const lgeo = new THREE.BufferGeometry();
  lgeo.setAttribute("position", new THREE.Float32BufferAttribute(linePts, 3));
  lgeo.setAttribute("color", new THREE.Float32BufferAttribute(lineCols, 3));
  const lmat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false });
  const lines = new THREE.LineSegments(lgeo, lmat);
  lines.geometry.setDrawRange(0, 0); // drawn progressively, neuron by neuron
  lines.userData.vertexCount = hidden.length * outputs.length * 2;
  group.add(lines);

  group.position.set(POS.fc.x, 0, POS.fc.z);
  return { group, hidden, outputs, lines };
}
