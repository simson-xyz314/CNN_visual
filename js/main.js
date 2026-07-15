import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { loadWeights, forward, getKernel } from "./cnn.js";
import { initPad } from "./pad.js";
import {
  tween, clearTweens, updateTweens, setGroupOpacity,
  makeTexture, easeInOut, easeOut, easeOutBack,
} from "./util.js";
import {
  POS, buildInput, buildRGB, buildKernel, buildFeatureLayer, buildFlatten, buildFC,
} from "./stages.js";

// ---------------- Scene setup ----------------
const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x05060d, 0.014);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 400);
camera.position.set(4, 5, 17);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enablePan = true;
controls.minDistance = 4;
controls.maxDistance = 140;
controls.enabled = false; // enabled after the flythrough

scene.add(new THREE.AmbientLight(0x8899cc, 0.65));
const dir = new THREE.DirectionalLight(0xffffff, 1.1);
dir.position.set(6, 12, 10);
scene.add(dir);
const pl1 = new THREE.PointLight(0x38e8ff, 0.8, 160); pl1.position.set(-12, 6, 6); scene.add(pl1);
const pl2 = new THREE.PointLight(0xffd23f, 0.5, 160); pl2.position.set(12, -6, -30); scene.add(pl2);

// Ambient drifting particles
const pGeo = new THREE.BufferGeometry();
const pcount = 600, parr = new Float32Array(pcount * 3);
let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
for (let i = 0; i < pcount; i++) {
  parr[i * 3] = (rnd() - 0.5) * 100;
  parr[i * 3 + 1] = (rnd() - 0.5) * 64;
  parr[i * 3 + 2] = (rnd() - 0.5) * 120 - 30;
}
pGeo.setAttribute("position", new THREE.Float32BufferAttribute(parr, 3));
const particles = new THREE.Points(pGeo, new THREE.PointsMaterial({ color: 0x4f7dff, size: 0.16, transparent: true, opacity: 0.45 }));
scene.add(particles);

const inputMat = new THREE.MeshStandardMaterial({ metalness: 0.2, roughness: 0.55, transparent: true, opacity: 1 });

// Forward-pass light beam
const beam = new THREE.Mesh(
  new THREE.SphereGeometry(0.55, 16, 16),
  new THREE.MeshBasicMaterial({ color: 0x9becff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })
);
beam.visible = false; scene.add(beam);
const beamGlow = new THREE.Mesh(
  new THREE.SphereGeometry(1.5, 16, 16),
  new THREE.MeshBasicMaterial({ color: 0x38e8ff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })
);
beam.add(beamGlow);
const _bA = new THREE.Vector3(), _bB = new THREE.Vector3();
function shootBeam(from, to) {
  _bA.set(from.x, 0, from.z); _bB.set(to.x, 0, to.z);
  beam.visible = true;
  tween({
    dur: 0.9, ease: easeOut,
    onUpdate: (e) => {
      beam.position.lerpVectors(_bA, _bB, e);
      const g = Math.sin(e * Math.PI);
      beam.material.opacity = g * 0.95; beamGlow.material.opacity = g * 0.4;
      beam.scale.setScalar(1 + g * 1.4);
    },
    onDone: () => { beam.visible = false; },
  });
}

// ---------------- Build the whole network ----------------
let B = null;

function disposeBuilt() {
  if (!B) return;
  for (const g of Object.values(B.groups)) scene.remove(g);
  B = null;
}

function buildAll(result) {
  disposeBuilt();
  const input = buildInput(result.input, inputMat.clone());
  const rgb = buildRGB(result.input);
  const kernel = buildKernel(getKernel("conv1", 0, 0));

  // scan-output preview plane placed to the right of the input
  const scanTex = makeTexture(result.conv1.data, 0, 28, 28, { mode: "cmap", signed: true });
  const scanMat = new THREE.MeshBasicMaterial({ map: scanTex, transparent: true, opacity: 0, side: THREE.DoubleSide });
  const scanPlane = new THREE.Mesh(new THREE.PlaneGeometry(28 * 0.5, 28 * 0.5), scanMat);
  const scanGroup = new THREE.Group(); scanGroup.add(scanPlane);
  scanGroup.position.set(POS.input.x + 17, 0, POS.input.z); // clear of the input grid (right edge at x=7)
  // faint frame scaffold for the scan output
  const scanFrame = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.PlaneGeometry(28 * 0.5, 28 * 0.5)),
    new THREE.LineBasicMaterial({ color: 0x33507f, transparent: true, opacity: 0.3 })
  );
  scanGroup.add(scanFrame);

  const conv1raw = buildFeatureLayer(result.conv1.data, 8, 28, 28, { cols: 4, cell: 0.14, gap: 0.8, pos: POS.conv1, signed: true });
  const conv1relu = buildFeatureLayer(result.conv1_relu.data, 8, 28, 28, { cols: 4, cell: 0.14, gap: 0.8, pos: { x: POS.conv1.x, z: POS.conv1.z - 0.35 } });
  const pool1 = buildFeatureLayer(result.pool1.data, 8, 14, 14, { cols: 4, cell: 0.28, gap: 0.8, pos: POS.pool1 });
  const conv2 = buildFeatureLayer(result.conv2_relu.data, 16, 14, 14, { cols: 8, cell: 0.2, gap: 0.5, pos: POS.conv2 });
  const pool2 = buildFeatureLayer(result.pool2.data, 16, 7, 7, { cols: 8, cell: 0.4, gap: 0.5, pos: POS.pool2 });
  const flatten = buildFlatten(result.flatten);
  const fc = buildFC(result.fc1, result.logits, result.probs);

  const groups = {
    input: input.group, rgb: rgb.group, kernel: kernel.group, scan: scanGroup,
    conv1raw: conv1raw.group, conv1relu: conv1relu.group, pool1: pool1.group,
    conv2: conv2.group, pool2: pool2.group, flatten: flatten.group, fc: fc.group,
  };
  for (const g of Object.values(groups)) { g.visible = true; scene.add(g); }
  input.mesh.scale.z = 0.001; // grow on reveal
  // transient groups start hidden
  rgb.group.userData._op = 0; setGroupOpacity(rgb.group, 0);
  kernel.group.userData._op = 0;
  kernel.cubes.forEach((c) => (c.material.opacity = 0));

  B = { result, input, rgb, kernel, scanPlane, scanMat, scanTex, conv1raw, conv1relu, pool1, conv2, pool2, flatten, fc, groups };
}

// reveal helpers (data stays visible once shown; scaffolds persist) --------
function fadeTo(group, to, dur = 0.7, delay = 0) {
  const start = group.userData._op ?? 0;
  group.userData._op = to;
  tween({ dur, delay, onUpdate: (e) => setGroupOpacity(group, start + (to - start) * e) });
}

function revealLayer(layer, { step = 0.06, pop = false, target = 1 } = {}) {
  layer.planes.forEach((pl, i) => {
    const s0 = pl.mat.opacity;
    if (pop) pl.sub.scale.setScalar(0.7);
    tween({
      dur: 0.6, delay: i * step, ease: pop ? easeOutBack : easeOut,
      onUpdate: (e) => {
        pl.mat.opacity = s0 + (target - s0) * Math.min(1, e);
        pl.slab.material.opacity = 0.14 + (0.9 - 0.14) * Math.min(1, e);
        if (pop) pl.sub.scale.setScalar(0.7 + 0.3 * Math.min(1, e));
      },
    });
  });
}
function dimLayerPlanes(layer, target = 0.15) {
  layer.planes.forEach((pl) => {
    const s0 = pl.mat.opacity;
    tween({ dur: 0.7, onUpdate: (e) => (pl.mat.opacity = s0 + (target - s0) * e) });
  });
}

// ---------------- Step metadata ----------------
const STEP_META = [
  ["손그림 업로드", "사용자가 그린 숫자를 28×28 격자의 흑백 이미지로 변환합니다. 각 칸(픽셀)의 밝기가 입력 데이터이며, 높이로 표현했습니다."],
  ["RGB 채널 분리", "컬러 이미지는 빨강·초록·파랑 3채널로 이루어집니다. 손글씨는 흑백이라 세 채널이 같고, 하나의 명암으로 합쳐 신경망에 넣습니다."],
  ["3×3 커널(필터)", "특징을 찾는 3×3 가중치 행렬입니다. 색은 학습된 가중치 값(양/음)을 나타냅니다. 이 커널이 이미지를 훑고 지나갑니다."],
  ["특징 추출 (합성곱)", "커널이 입력 위를 한 칸씩 미끄러지며 겹친 부분을 곱해 더합니다. 오른쪽 특징맵이 스캔과 동기화되어 채워집니다."],
  ["피처 맵 생성", "커널 8개가 각기 다른 특징(가장자리·곡선 등)을 잡아 8장의 특징맵을 만듭니다. 밝을수록 그 특징이 강한 위치입니다."],
  ["ReLU 활성화", "음수는 모두 0으로 만들고 양수만 남깁니다. 중요한 특징만 통과시켜 비선형성을 부여합니다. 어두워진 칸이 0으로 잘린 부분입니다."],
  ["풀링 (Pooling)", "2×2 영역에서 최댓값만 남겨 28×28 → 14×14로 축소합니다. 위치 변화에 강해지고 계산량이 줄어듭니다."],
  ["합성곱 반복", "앞 특징맵에 커널 16개를 다시 적용합니다. 얕은 특징(선)을 조합해 더 복잡한 패턴을 학습합니다."],
  ["깊은 특징 추출", "두 번째 풀링까지 거친 16장의 7×7 특징맵. 숫자를 구분하는 고수준의 추상적 특징이 담깁니다."],
  ["Flatten (평탄화)", "16×7×7 = 784개 값을 한 줄의 벡터로 펼칩니다. 3차원 특징을 완전연결층이 받도록 1차원으로 바꿉니다."],
  ["완전연결층 (FC)", "784개 값이 64개 뉴런과, 다시 10개 출력 뉴런과 연결됩니다. 연결선이 밝을수록 기여가 큽니다."],
  ["Softmax", "10개 출력값을 합이 1인 확률로 변환합니다. 가장 큰 값을 갖는 숫자가 최종 예측입니다."],
  ["결과 출력", "각 숫자(0~9)일 확률을 막대그래프로 보여줍니다. 가장 높은 숫자가 모델의 예측입니다."],
];

// Camera waypoints per step (world space). Stages run left->right along X, so
// the camera pans sideways; earlier layers sit to the left and never occlude.
const cx = { input: POS.input.x, conv1: POS.conv1.x, pool1: POS.pool1.x, conv2: POS.conv2.x, pool2: POS.pool2.x, flatten: POS.flatten.x, fc: POS.fc.x };
const cz = { input: POS.input.z, conv1: POS.conv1.z, pool1: POS.pool1.z, conv2: POS.conv2.z, pool2: POS.pool2.z, flatten: POS.flatten.z, fc: POS.fc.z };
const WP = [
  { p: [cx.input - 3, 5, 17], t: [cx.input, 0.6, 0] },                 // 1 input
  { p: [cx.input, 2.5, 15], t: [cx.input, 0, 0] },                     // 2 rgb
  { p: [cx.input - 2, 4, 12], t: [cx.input, 1, 0.5] },                 // 3 kernel
  { p: [cx.input + 8, 5, 23], t: [cx.input + 8, 0, -2] },             // 4 scan
  { p: [cx.conv1, 2, cz.conv1 + 19], t: [cx.conv1, 0, cz.conv1] },     // 5 maps
  { p: [cx.conv1, 2, cz.conv1 + 17], t: [cx.conv1, 0, cz.conv1] },     // 6 relu
  { p: [cx.pool1, 2, cz.pool1 + 19], t: [cx.pool1, 0, cz.pool1] },     // 7 pool1
  { p: [cx.conv2, 2.5, cz.conv2 + 30], t: [cx.conv2, 0, cz.conv2] },   // 8 conv2
  { p: [cx.pool2, 2.5, cz.pool2 + 30], t: [cx.pool2, 0, cz.pool2] },   // 9 pool2
  { p: [cx.flatten, 2, cz.flatten + 23], t: [cx.flatten, 0, cz.flatten] }, // 10 flatten
  { p: [cx.fc + 3, 2.5, cz.fc + 30], t: [cx.fc + 5, 0, cz.fc] },       // 11 fc
  { p: [cx.fc + 8, 2, cz.fc + 18], t: [cx.fc + 10, 0, cz.fc] },        // 12 softmax
  { p: [cx.fc + 9, 1.5, cz.fc + 15], t: [cx.fc + 10, 0, cz.fc] },      // 13 output
];
const WPp = WP.map((w) => new THREE.Vector3(...w.p));
const WPt = WP.map((w) => new THREE.Vector3(...w.t));
const _tgt = new THREE.Vector3();

// Each step = HOLD (camera parked at the layer while its reveal animation plays
// to completion) + TRAVEL (glide to the next layer). This guarantees an
// animation finishes before the next one starts.
const HOLD = [2.2, 3.8, 1.8, 6.2, 2.4, 2.2, 2.2, 2.4, 2.2, 2.6, 2.8, 2.2, 4.2];
const TRAVEL = 1.7;
const N = 13;
const stepStart = [0];
for (let i = 1; i < N; i++) stepStart[i] = stepStart[i - 1] + HOLD[i - 1] + TRAVEL;
const totalTime = stepStart[N - 1] + HOLD[N - 1];

// ---------------- Reveal behaviours per step ----------------
function reveal(step) {
  switch (step) {
    case 1:
      fadeTo(B.groups.input, 1, 0.5);
      tween({ dur: 1.2, ease: easeOutBack, onUpdate: (e) => (B.input.mesh.scale.z = Math.max(0.001, e)) });
      break;
    case 2:
      fadeTo(B.groups.rgb, 0.9, 0.5);
      B.rgb.planes.forEach((pl) => (pl.material.opacity = 0));
      tween({ dur: 0.5, onUpdate: (e) => B.rgb.planes.forEach((pl) => (pl.material.opacity = e * 0.9)) });
      tween({
        dur: 3.0, delay: 0.3, ease: (x) => x,
        onUpdate: (e) => {
          const off = Math.sin(e * Math.PI) * 3.6;
          B.rgb.planes[0].position.set(-off, 0, Math.sin(e * Math.PI) * 0.6);
          B.rgb.planes[2].position.set(off, 0, -Math.sin(e * Math.PI) * 0.6);
        },
      });
      break;
    case 3:
      fadeTo(B.groups.rgb, 0, 0.6);
      B.kernel.cubes.forEach((c, i) => tween({ dur: 0.5, delay: i * 0.03, ease: easeOutBack, onUpdate: (e) => (c.material.opacity = e) }));
      B.groups.kernel.position.set(POS.input.x - 6.75, 6.75, POS.input.z + 3.4);
      break;
    case 4:
      B.scanMat.opacity = 1;
      startScan();
      break;
    case 5:
      fadeTo(B.groups.kernel, 0, 0.6);
      B.kernel.hlMat.opacity = 0;
      tween({ dur: 0.6, onUpdate: (e) => (B.scanMat.opacity = 1 - e), onDone: () => (B.scanMat.opacity = 0) });
      shootBeam(POS.input, POS.conv1);
      revealLayer(B.conv1raw, { pop: true });
      break;
    case 6:
      // in-place value change: raw -> ReLU (local crossfade, camera stays put)
      dimLayerPlanes(B.conv1raw, 0.0);
      revealLayer(B.conv1relu, { step: 0.04 });
      break;
    case 7: shootBeam(POS.conv1, POS.pool1); revealLayer(B.pool1, { pop: true }); break;
    case 8: shootBeam(POS.pool1, POS.conv2); revealLayer(B.conv2, { step: 0.04 }); break;
    case 9: shootBeam(POS.conv2, POS.pool2); revealLayer(B.pool2, { pop: true }); break;
    case 10:
      shootBeam(POS.pool2, POS.flatten);
      B.flatten.mesh.material.opacity = 0;
      B.flatten.group.scale.x = 0.12;
      tween({ dur: 0.6, onUpdate: (e) => (B.flatten.mesh.material.opacity = e) });
      tween({ dur: 1.5, ease: easeOut, onUpdate: (e) => (B.flatten.group.scale.x = 0.12 + 0.88 * e) });
      break;
    case 11: shootBeam(POS.flatten, POS.fc); enterFC(); break;
    case 12: pulseSoftmax(); break;
    case 13: showResult(); break;
  }
}

function startScan() {
  const kz = POS.input.z + 3.4;
  // serpentine (boustrophedon) visiting order over the 28x28 output
  const order = [];
  for (let gy = 0; gy < 28; gy++) {
    if (gy % 2 === 0) for (let gx = 0; gx < 28; gx++) order.push(gy * 28 + gx);
    else for (let gx = 27; gx >= 0; gx--) order.push(gy * 28 + gx);
  }
  const buf = B.scanTex._buf;
  for (let i = 0; i < 784; i++) buf[i * 4 + 3] = 0;
  B.scanTex.needsUpdate = true;
  const cellToWorld = (idx) => {
    const gx = idx % 28, gy = (idx / 28) | 0;
    return [POS.input.x - 7 + (gx + 0.5) * 0.5, 7 - (gy + 0.5) * 0.5];
  };
  tween({
    dur: 5.6, ease: (x) => x,
    onUpdate: (e) => {
      const kf = e * 783;
      const k = Math.min(783, Math.floor(kf));
      // reveal all visited cells
      for (let i = 0; i <= k; i++) buf[order[i] * 4 + 3] = 255;
      B.scanTex.needsUpdate = true;
      // smooth kernel position between current and next visited cell
      const [ax, ay] = cellToWorld(order[k]);
      const [bx, by] = cellToWorld(order[Math.min(783, k + 1)]);
      const f = kf - k;
      const wx = ax + (bx - ax) * f, wy = ay + (by - ay) * f;
      B.groups.kernel.position.set(wx, wy, kz);
      B.kernel.hlMat.opacity = 0.28;
      B.kernel.hl.position.set(wx, wy, 0.06);
      if (!B.kernel.hl.parent) B.groups.input.add(B.kernel.hl);
    },
    onDone: () => { for (let i = 0; i < 784; i++) buf[i * 4 + 3] = 255; B.scanTex.needsUpdate = true; },
  });
}

function enterFC() {
  B.fc.hidden.forEach((h, i) => { h.mesh.material.opacity = 0; tween({ dur: 0.5, delay: i * 0.008, onUpdate: (e) => (h.mesh.material.opacity = e) }); });
  B.fc.outputs.forEach((o, i) => { o.mesh.material.opacity = 0; tween({ dur: 0.5, delay: 1.4 + i * 0.04, onUpdate: (e) => (o.mesh.material.opacity = e) }); });
  // draw the connection web progressively, neuron by neuron (left -> right)
  const lines = B.fc.lines;
  const total = lines.userData.vertexCount;
  lines.material.opacity = 0.5;
  lines.geometry.setDrawRange(0, 0);
  tween({
    dur: 1.9, delay: 0.5, ease: easeOut,
    onUpdate: (e) => {
      const perNeuron = 20; // 10 outputs * 2 verts
      const v = Math.floor((e * total) / perNeuron) * perNeuron; // reveal full neuron fans
      lines.geometry.setDrawRange(0, Math.min(total, v));
    },
    onDone: () => lines.geometry.setDrawRange(0, total),
  });
}

function pulseSoftmax() {
  B.fc.outputs.forEach((o) => {
    const target = 0.6 + o.prob * 2.4;
    tween({ dur: 0.9, delay: 0.15, ease: easeOutBack, onUpdate: (e) => o.mesh.scale.setScalar(1 + (target - 1) * e) });
  });
}

// ---------------- D3 probability chart ----------------
function showResult() {
  const probs = B.result.probs, pred = B.result.prediction;
  document.getElementById("result-digit").textContent = pred;
  document.getElementById("result-conf").textContent = `확신도 ${(probs[pred] * 100).toFixed(1)}%`;
  document.getElementById("result-card").classList.remove("hidden");

  const toast = document.getElementById("toast");
  toast.innerHTML = `당신이 쓴 숫자는 <b>${pred}</b> 입니다`;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2800);

  const svg = d3.select("#prob-chart");
  svg.selectAll("*").remove();
  const W = 360, H = 300, m = { l: 26, r: 44, t: 8, b: 8 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const y = d3.scaleBand().domain(d3.range(10)).range([m.t, m.t + ih]).padding(0.22);
  const x = d3.scaleLinear().domain([0, 1]).range([0, iw]);
  const g = svg.append("g").attr("transform", `translate(${m.l},0)`);
  g.selectAll(".lbl").data(d3.range(10)).enter().append("text")
    .attr("x", -8).attr("y", (d) => y(d) + y.bandwidth() / 2 + 4).attr("text-anchor", "end").attr("class", "bar-label")
    .style("font-weight", (d) => (d === pred ? 800 : 400)).style("fill", (d) => (d === pred ? "#38e8ff" : "#8ea2c7")).text((d) => d);
  const bars = g.selectAll(".bar").data(d3.range(10)).enter().append("rect")
    .attr("x", 0).attr("y", (d) => y(d)).attr("height", y.bandwidth()).attr("rx", 5)
    .attr("fill", (d) => (d === pred ? "#38e8ff" : "#2a3a66")).attr("width", 0);
  bars.transition().duration(900).delay((d) => d * 40).ease(d3.easeCubicOut).attr("width", (d) => Math.max(2, x(probs[d])));
  g.selectAll(".val").data(d3.range(10)).enter().append("text")
    .attr("x", (d) => x(probs[d]) + 6).attr("y", (d) => y(d) + y.bandwidth() / 2 + 4).attr("class", "bar-val").style("opacity", 0)
    .text((d) => `${(probs[d] * 100).toFixed(0)}%`).transition().delay((d) => 500 + d * 40).duration(500).style("opacity", 1);
}

// ---------------- Flow controller ----------------
let flowActive = false, flowTime = 0, lastStep = -1, flowFrozen = false;

function startFlow() {
  clearTweens();
  flowActive = true; flowTime = 0; lastStep = 0;
  controls.enabled = false;
  document.getElementById("progress").classList.remove("hidden");
  document.getElementById("result-card").classList.add("hidden");
}

// ---------------- UI wiring ----------------
let pad = null;

async function boot() {
  await loadWeights("model_weights.json");
  pad = initPad();
  document.getElementById("loading").classList.add("done");
  document.getElementById("clear-btn").onclick = () => pad.clear();
  document.getElementById("run-btn").onclick = runInference;
  window.__viz = {
    ready: true, runInput, totalTime, stepStart, HOLD,
    jump: (step) => { flowTime = Math.min(totalTime, stepStart[step - 1] + HOLD[step - 1] * 0.85); },
    jumpFrac: (step, frac) => { flowTime = stepStart[step - 1] + HOLD[step - 1] * frac; },
  };
}

function runInference() {
  if (!pad.hasInk()) { flashPad(); return; }
  const input = pad.extract();
  if (!input) { flashPad(); return; }
  runInput(input);
}

function runInput(input) {
  const result = forward(input);
  buildAll(result);
  document.getElementById("pad-panel").classList.add("dim");
  startFlow();
}

function flashPad() {
  const p = document.getElementById("pad");
  p.animate([{ boxShadow: "0 0 0 2px #ff5577" }, { boxShadow: "inset 0 0 40px rgba(0,0,0,.6)" }], { duration: 600 });
}

// ---------------- Render loop ----------------
let lastT = 0;
function animate(t) {
  requestAnimationFrame(animate);
  const now = t / 1000;
  const dt = Math.min(0.05, now - lastT || 0.016);
  lastT = now;
  updateTweens(dt);
  particles.rotation.y += dt * 0.02;

  if (flowActive && B) {
    if (!flowFrozen) flowTime = Math.min(flowTime + dt, totalTime);
    let s = 0;
    while (s < N - 1 && flowTime >= stepStart[s + 1]) s++;
    const local = flowTime - stepStart[s];
    // fire every step reveal up to the current one (robust to seeking)
    while (lastStep < s + 1) { lastStep++; reveal(lastStep); }
    // HOLD: park exactly at this layer; TRAVEL: glide to the next waypoint
    const next = Math.min(N - 1, s + 1);
    const fr = (local < HOLD[s] || s >= N - 1) ? 0 : easeInOut(Math.min(1, (local - HOLD[s]) / TRAVEL));
    camera.position.lerpVectors(WPp[s], WPp[next], fr);
    _tgt.lerpVectors(WPt[s], WPt[next], fr);
    controls.target.copy(_tgt);
    camera.lookAt(_tgt);
    document.getElementById("progress-fill").style.width = `${(flowTime / totalTime) * 100}%`;
    if (flowTime >= totalTime) { flowActive = false; controls.enabled = true; }
  } else if (controls.enabled) {
    controls.update();
  }

  if (B) B.fc.outputs.forEach((o) => (o.mesh.material.emissiveIntensity = 0.15 + o.prob * (1.2 + Math.sin(now * 2 + o.digit) * 0.25)));
  renderer.render(scene, camera);
}

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

boot();
requestAnimationFrame(animate);
