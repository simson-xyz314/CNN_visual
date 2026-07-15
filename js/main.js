import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { loadWeights, forward, getKernel } from "./cnn.js";
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

// ---------------- Floating station labels (small 3D captions parked above each layer) ----------------
function makeLabelTexture(text) {
  const dpr = 2, fs = 46;
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  ctx.font = `700 ${fs}px 'Malgun Gothic', sans-serif`;
  const tw = ctx.measureText(text).width;
  c.width = (tw + 70) * dpr; c.height = (fs + 44) * dpr;
  ctx.scale(dpr, dpr);
  const w = c.width / dpr, h = c.height / dpr;
  ctx.font = `700 ${fs}px 'Malgun Gothic', sans-serif`;
  ctx.fillStyle = "rgba(10,16,34,0.72)";
  if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(3, 3, w - 6, h - 6, 18); ctx.fill(); }
  else ctx.fillRect(3, 3, w - 6, h - 6);
  ctx.strokeStyle = "rgba(120,160,255,0.35)"; ctx.lineWidth = 2; ctx.stroke();
  ctx.fillStyle = "#eaf2ff"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.shadowColor = "rgba(56,232,255,0.5)"; ctx.shadowBlur = 12;
  ctx.fillText(text, w / 2, h / 2 + 2);
  const tex = new THREE.CanvasTexture(c);
  tex.anisotropy = 4;
  return { tex, aspect: w / h };
}

function makeLabelSprite(text) {
  const { tex, aspect } = makeLabelTexture(text);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0, depthTest: false, depthWrite: false }));
  sp.renderOrder = 999; sp.userData.aspect = aspect;
  return sp;
}

// Station signs parked above the layers of the fast middle section. They stay
// hidden until the train reaches that section, then appear in advance so you can
// read them while whizzing past.
let stationLabels = {};
function buildStationLabels() {
  for (const k in stationLabels) scene.remove(stationLabels[k]);
  stationLabels = {};
  const defs = {
    conv1: { x: POS.conv1.x, y: 6.8, z: POS.conv1.z, t: "피처맵 · ReLU" },
    pool1: { x: POS.pool1.x, y: 6.8, z: POS.pool1.z, t: "풀링" },
    conv2: { x: POS.conv2.x, y: 5.6, z: POS.conv2.z, t: "합성곱 반복" },
    pool2: { x: POS.pool2.x, y: 5.6, z: POS.pool2.z, t: "깊은 특징" },
    flatten: { x: POS.flatten.x, y: 5.0, z: POS.flatten.z, t: "Flatten" },
    fc: { x: POS.fc.x + 5, y: 7.6, z: POS.fc.z, t: "완전연결 · Softmax · 예측" },
  };
  for (const k in defs) {
    const d = defs[k];
    const sp = makeLabelSprite(d.t);
    sp.position.set(d.x, d.y, d.z);
    scene.add(sp);
    stationLabels[k] = sp;
  }
}
function showStation(...keys) {
  for (const k of keys) {
    const sp = stationLabels[k];
    if (!sp) continue;
    const s0 = sp.material.opacity;
    tween({ dur: 0.5, onUpdate: (e) => (sp.material.opacity = s0 + (1 - s0) * e) });
  }
}

// A single per-step caption used for the early distinct steps (1~4), which all
// happen at the input location. Fades out once the station signs take over.
const captionSprite = new THREE.Sprite(new THREE.SpriteMaterial({ transparent: true, opacity: 0, depthTest: false, depthWrite: false }));
captionSprite.renderOrder = 999; captionSprite.visible = false; captionSprite.userData.aspect = 4;
scene.add(captionSprite);
function setCaption(text, pos) {
  const { tex, aspect } = makeLabelTexture(text);
  if (captionSprite.material.map) captionSprite.material.map.dispose();
  captionSprite.material.map = tex; captionSprite.material.needsUpdate = true;
  captionSprite.userData.aspect = aspect;
  captionSprite.position.set(pos[0], pos[1], pos[2]);
  captionSprite.visible = true;
  const s0 = captionSprite.material.opacity;
  tween({ dur: 0.5, onUpdate: (e) => (captionSprite.material.opacity = s0 + (1 - s0) * e) });
}
function hideCaption() {
  const s0 = captionSprite.material.opacity;
  tween({ dur: 0.4, onUpdate: (e) => (captionSprite.material.opacity = s0 * (1 - e)), onDone: () => (captionSprite.visible = false) });
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
// Oblique "train window" view: the camera sits behind-left of each layer and
// looks forward-right, so it glides past the layers at an angle (지하철 차창).
const WP = [
  { p: [cx.input - 6, 5, 17], t: [cx.input + 3, 0.6, 0] },                       // 1 input
  { p: [cx.input - 11, 4, 13], t: [cx.input, 0, 0] },                            // 2 rgb (angled to see the 3 channels)
  { p: [cx.input - 6, 4.5, 12], t: [cx.input + 2, 1, 0.5] },                     // 3 kernel
  { p: [cx.input + 2, 5, 22], t: [cx.input + 10, 0, -2] },                       // 4 scan
  { p: [cx.conv1 - 9, 4, cz.conv1 + 15], t: [cx.conv1 + 4, 0, cz.conv1] },       // 5 maps
  { p: [cx.conv1 - 6, 4, cz.conv1 + 13], t: [cx.conv1 + 6, 0, cz.conv1] },       // 6 relu
  { p: [cx.pool1 - 9, 4, cz.pool1 + 15], t: [cx.pool1 + 4, 0, cz.pool1] },       // 7 pool1
  { p: [cx.conv2 - 12, 4.5, cz.conv2 + 22], t: [cx.conv2 + 5, 0, cz.conv2] },    // 8 conv2
  { p: [cx.pool2 - 12, 4.5, cz.pool2 + 22], t: [cx.pool2 + 5, 0, cz.pool2] },    // 9 pool2
  { p: [cx.flatten - 9, 4, cz.flatten + 17], t: [cx.flatten + 4, 0, cz.flatten] }, // 10 flatten
  { p: [cx.fc - 7, 4.5, cz.fc + 24], t: [cx.fc + 6, 0, cz.fc] },                 // 11 fc
  { p: [cx.fc + 2, 3, cz.fc + 16], t: [cx.fc + 10, 0, cz.fc] },                  // 12 softmax
  { p: [cx.fc + 4, 2.5, cz.fc + 14], t: [cx.fc + 10, 0, cz.fc] },                // 13 output
];
const WPp = WP.map((w) => new THREE.Vector3(...w.p));
const WPt = WP.map((w) => new THREE.Vector3(...w.t));
const _tgt = new THREE.Vector3();

// Each step = HOLD (camera dwells at the layer while its reveal plays) + TRAVEL
// (glide to the next). Visually-similar middle stages (ReLU, pooling, repeated
// conv) get short HOLD/TRAVEL so the train zips through them; distinctive stages
// (input, scan, first maps, flatten, FC, output) linger.
// Steps 1-5 and 10-13 keep the original pace; only the visually-similar middle
// (6 ReLU, 7 pool, 8 conv, 9 pool) whizzes by fast.
//              1    2    3    4    5    6    7    8    9   10   11   12   13
const HOLD =   [2.2, 3.8, 1.8, 6.2, 2.4, 0.9, 1.0, 1.2, 0.9, 2.6, 2.8, 2.2, 4.2];
const TRAVEL = [1.7, 1.7, 1.7, 1.7, 1.2, 0.8, 0.9, 0.8, 1.6, 1.7, 1.7, 1.7, 0.0];
const N = 13;
const stepStart = [0];
for (let i = 1; i < N; i++) stepStart[i] = stepStart[i - 1] + HOLD[i - 1] + TRAVEL[i - 1];
const totalTime = stepStart[N - 1] + HOLD[N - 1];

// ---------------- Reveal behaviours per step ----------------
function reveal(step) {
  switch (step) {
    case 1:
      setCaption("1. 입력 이미지 (28×28)", [POS.input.x, 9.5, 0]);
      fadeTo(B.groups.input, 1, 0.5);
      tween({ dur: 1.2, ease: easeOutBack, onUpdate: (e) => (B.input.mesh.scale.z = Math.max(0.001, e)) });
      break;
    case 2:
      setCaption("2. RGB 채널 분리", [POS.input.x, 9.5, 0]);
      fadeTo(B.groups.input, 0.06, 0.6); // background grid nearly transparent while RGB planes show
      fadeTo(B.groups.rgb, 0.9, 0.5);
      B.rgb.planes.forEach((pl) => (pl.material.opacity = 0));
      tween({ dur: 0.5, onUpdate: (e) => B.rgb.planes.forEach((pl) => (pl.material.opacity = e * 0.9)) });
      // pull the three channels apart into clearly spaced layers, then HOLD (no merge)
      tween({
        dur: 1.3, delay: 0.3, ease: easeOut,
        onUpdate: (e) => {
          B.rgb.planes[0].position.set(-1.3 * e, 0.9 * e, 3.2 * e);   // R — front
          B.rgb.planes[1].position.set(0, 0, 0);                      // G — middle
          B.rgb.planes[2].position.set(1.3 * e, -0.9 * e, -3.2 * e);  // B — back
        },
      });
      break;
    case 3:
      setCaption("3. 3×3 커널", [POS.input.x, 9.5, 0]);
      fadeTo(B.groups.input, 1, 0.6); // restore the input grid for the scan
      fadeTo(B.groups.rgb, 0, 0.6);
      B.kernel.cubes.forEach((c, i) => tween({ dur: 0.5, delay: i * 0.03, ease: easeOutBack, onUpdate: (e) => (c.material.opacity = e) }));
      B.groups.kernel.position.set(POS.input.x - 6.75, 6.75, POS.input.z + 3.4);
      break;
    case 4:
      setCaption("4. 특징 추출 (합성곱)", [POS.input.x + 8, 9.5, -1]);
      B.scanMat.opacity = 1;
      startScan();
      break;
    case 5:
      fadeTo(B.groups.kernel, 0, 0.6);
      B.kernel.hlMat.opacity = 0;
      tween({ dur: 0.6, onUpdate: (e) => (B.scanMat.opacity = 1 - e), onDone: () => (B.scanMat.opacity = 0) });
      hideCaption(); // station signs take over from here
      shootBeam(POS.input, POS.conv1);
      revealLayer(B.conv1raw, { pop: true });
      showStation("conv1", "pool1", "conv2", "pool2"); // whiz-section signs appear in advance
      break;
    case 6:
      // in-place value change: raw -> ReLU (local crossfade, camera stays put)
      dimLayerPlanes(B.conv1raw, 0.0);
      revealLayer(B.conv1relu, { step: 0.04 });
      break;
    case 7: shootBeam(POS.conv1, POS.pool1); revealLayer(B.pool1, { pop: true }); break;
    case 8: shootBeam(POS.pool1, POS.conv2); revealLayer(B.conv2, { step: 0.04 }); break;
    case 9: shootBeam(POS.conv2, POS.pool2); revealLayer(B.pool2, { pop: true }); showStation("flatten", "fc"); break;
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

  // ~10s after the result appears, clear the scene back to the idle screen
  clearTimeout(idleTimer);
  idleTimer = setTimeout(resetView, 10000);

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
let flowActive = false, flowTime = 0, lastStep = -1, flowFrozen = false, idleTimer = null;

function startFlow() {
  clearTweens();
  clearTimeout(idleTimer);
  flowActive = true; flowTime = 0; lastStep = 0;
  controls.enabled = false;
  captionSprite.visible = false; captionSprite.material.opacity = 0;
  buildStationLabels();
  document.getElementById("progress").classList.remove("hidden");
  document.getElementById("result-card").classList.add("hidden");
}

// ---------------- Display: receive digits from the pad over WebSocket ----------------
async function boot() {
  await loadWeights("model_weights.json");
  document.getElementById("loading").classList.add("done");
  connectWS();
  window.__viz = {
    ready: true, runInput, totalTime, stepStart, HOLD,
    freeze: (v) => { flowFrozen = v; },
    jump: (step) => { flowTime = Math.min(totalTime, stepStart[step - 1] + HOLD[step - 1] * 0.85); },
    jumpFrac: (step, frac) => { flowTime = stepStart[step - 1] + HOLD[step - 1] * frac; },
  };
}

function connectWS() {
  const conn = document.getElementById("wait-conn");
  const ws = new WebSocket(`ws://${location.host}`);
  ws.onopen = () => { ws.send(JSON.stringify({ type: "hello", role: "display" })); conn.textContent = "태블릿에 숫자를 그려서 시작하기"; conn.className = "connected"; };
  ws.onclose = () => { conn.textContent = "태블릿에 숫자를 그려서 시작하기"; conn.className = "disconnected"; setTimeout(connectWS, 1500); };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    let msg; try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "run" && Array.isArray(msg.input)) runInput(new Float32Array(msg.input));
    else if (msg.type === "clear") resetView();
  };
}

function runInput(input) {
  const result = forward(input);
  buildAll(result);
  document.getElementById("wait-card").classList.add("hidden");
  startFlow();
}

function resetView() {
  clearTimeout(idleTimer);
  flowActive = false;
  controls.enabled = false;
  captionSprite.visible = false; captionSprite.material.opacity = 0;
  for (const k in stationLabels) scene.remove(stationLabels[k]);
  stationLabels = {};
  document.getElementById("progress").classList.add("hidden");
  document.getElementById("result-card").classList.add("hidden");
  document.getElementById("wait-card").classList.remove("hidden");
  disposeBuilt();
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
    const fr = (local < HOLD[s] || s >= N - 1) ? 0 : easeInOut(Math.min(1, (local - HOLD[s]) / TRAVEL[s]));
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
  const scaleLabel = (sp) => {
    const d = camera.position.distanceTo(sp.position);
    const h = Math.max(0.5, d * 0.045); // keep captions roughly constant on-screen size
    sp.scale.set(h * sp.userData.aspect, h, 1);
  };
  for (const k in stationLabels) scaleLabel(stationLabels[k]);
  if (captionSprite.visible) scaleLabel(captionSprite);
  renderer.render(scene, camera);
}

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

boot();
requestAnimationFrame(animate);
