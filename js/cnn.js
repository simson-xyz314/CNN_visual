// Pure-JS forward pass mirroring the PyTorch SmallCNN in train_mnist.py.
// Produces every intermediate activation so the 3D visualization can show the
// real data flowing through the network for whatever digit the user drew.

// A tensor is { data: Float32Array, shape: [C, H, W] } (or [N] for vectors).

let W = null; // weights object from model_weights.json

export async function loadWeights(url = "model_weights.json") {
  const res = await fetch(url);
  const json = await res.json();
  W = json.weights;
  W.__shapes = json.shapes;
  return W;
}

function conv2d(input, weight, bias, outCh, inCh, k = 3, pad = 1) {
  const [C, H, Wd] = input.shape;
  const out = new Float32Array(outCh * H * Wd);
  for (let o = 0; o < outCh; o++) {
    const b = bias[o];
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < Wd; x++) {
        let acc = b;
        for (let i = 0; i < inCh; i++) {
          for (let ky = 0; ky < k; ky++) {
            const iy = y + ky - pad;
            if (iy < 0 || iy >= H) continue;
            for (let kx = 0; kx < k; kx++) {
              const ix = x + kx - pad;
              if (ix < 0 || ix >= Wd) continue;
              const w = weight[((o * inCh + i) * k + ky) * k + kx];
              acc += w * input.data[(i * H + iy) * Wd + ix];
            }
          }
        }
        out[(o * H + y) * Wd + x] = acc;
      }
    }
  }
  return { data: out, shape: [outCh, H, Wd] };
}

function relu(t) {
  const out = new Float32Array(t.data.length);
  for (let i = 0; i < t.data.length; i++) out[i] = Math.max(0, t.data[i]);
  return { data: out, shape: t.shape.slice() };
}

function maxpool2(t) {
  const [C, H, Wd] = t.shape;
  const oH = Math.floor(H / 2), oW = Math.floor(Wd / 2);
  const out = new Float32Array(C * oH * oW);
  for (let c = 0; c < C; c++) {
    for (let y = 0; y < oH; y++) {
      for (let x = 0; x < oW; x++) {
        let m = -Infinity;
        for (let dy = 0; dy < 2; dy++)
          for (let dx = 0; dx < 2; dx++) {
            const v = t.data[(c * H + (y * 2 + dy)) * Wd + (x * 2 + dx)];
            if (v > m) m = v;
          }
        out[(c * oH + y) * oW + x] = m;
      }
    }
  }
  return { data: out, shape: [C, oH, oW] };
}

function linear(vec, weight, bias, outN, inN) {
  const out = new Float32Array(outN);
  for (let o = 0; o < outN; o++) {
    let acc = bias[o];
    for (let i = 0; i < inN; i++) acc += weight[o * inN + i] * vec[i];
    out[o] = acc;
  }
  return out;
}

function softmax(vec) {
  const m = Math.max(...vec);
  const exps = vec.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

// input28: Float32Array length 784 in [0,1], shape 28x28 (single channel)
export function forward(input28) {
  const input = { data: input28, shape: [1, 28, 28] };

  const c1 = conv2d(input, W["conv1.weight"], W["conv1.bias"], 8, 1);
  const c1r = relu(c1);
  const p1 = maxpool2(c1r);

  const c2 = conv2d(p1, W["conv2.weight"], W["conv2.bias"], 16, 8);
  const c2r = relu(c2);
  const p2 = maxpool2(c2r);

  const flat = p2.data; // length 16*7*7 = 784

  const f1 = linear(flat, W["fc1.weight"], W["fc1.bias"], 64, 784);
  const f1r = f1.map((v) => Math.max(0, v));
  const logits = linear(f1r, W["fc2.weight"], W["fc2.bias"], 10, 64);
  const probs = softmax(logits);

  return {
    input, // [1,28,28]
    conv1: c1, // [8,28,28]  raw feature maps
    conv1_relu: c1r, // [8,28,28]
    pool1: p1, // [8,14,14]
    conv2: c2, // [16,14,14]
    conv2_relu: c2r, // [16,14,14]
    pool2: p2, // [16,7,7]
    flatten: flat, // [784]
    fc1: f1r, // [64]
    logits, // [10]
    probs, // [10]
    prediction: probs.indexOf(Math.max(...probs)),
  };
}

export function getKernel(layer, outCh, inCh) {
  // returns 3x3 kernel weights for display
  const key = layer + ".weight";
  const w = W[key];
  const inTotal = layer === "conv1" ? 1 : 8;
  const base = ((outCh * inTotal + inCh) * 3) * 3;
  return Array.from({ length: 9 }, (_, i) => w[base + i]);
}
