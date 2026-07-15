// Mobile drawing pad. Draws locally and sends the 28x28 digit to the display
// (laptop) over WebSocket on the same WiFi.
import { initPad } from "./pad.js";

// Enter fullscreen on the first user interaction (browsers block auto-fullscreen
// without a gesture). The first tap to draw triggers it.
function goFullscreen() {
  const el = document.documentElement;
  const rf = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen;
  if (rf) { try { rf.call(el).catch(() => {}); } catch {} }
}
["pointerdown", "touchend", "click"].forEach((ev) =>
  window.addEventListener(ev, goFullscreen, { once: true, capture: true })
);

const pad = initPad();
const status = document.getElementById("conn-status");
const toast = document.getElementById("pad-toast");

let ws = null;
function connect() {
  ws = new WebSocket(`ws://${location.host}`);
  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "hello", role: "pad" }));
    status.textContent = "노트북과 연결됨";
    status.className = "connected";
  };
  ws.onclose = () => {
    status.textContent = "연결 끊김 · 재연결 중…";
    status.className = "disconnected";
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
}
connect();

function flash(msg) {
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 1400);
}

document.getElementById("clear-btn").onclick = () => {
  pad.clear();
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "clear" }));
};

document.getElementById("run-btn").onclick = () => {
  if (!pad.hasInk()) { flash("숫자를 먼저 써주세요"); return; }
  const input = pad.extract();
  if (!input) { flash("숫자를 먼저 써주세요"); return; }
  if (!ws || ws.readyState !== WebSocket.OPEN) { flash("노트북과 연결 안 됨"); return; }
  ws.send(JSON.stringify({ type: "run", input: Array.from(input) }));
  flash("노트북 화면으로 전송됨!");
};
