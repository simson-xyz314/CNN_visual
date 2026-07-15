// CNN 3D 시각화 — 패드(휴대폰)와 디스플레이(노트북) 분리 서버.
// 같은 와이파이(LAN)에서:
//   노트북 화면 :  http://<이 PC의 IP>:8000/
//   휴대폰 패드 :  http://<이 PC의 IP>:8000/pad
// 휴대폰에서 그린 숫자를 WebSocket으로 노트북 화면에 실시간 전달합니다.

const http = require("http");
const fs = require("fs");
const path = require("path");
const os = require("os");
const { WebSocketServer } = require("ws");

const PORT = process.env.PORT || 8000;
const ROOT = __dirname;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".ico": "image/x-icon",
};

const server = http.createServer((req, res) => {
  let urlPath = decodeURIComponent(req.url.split("?")[0]);
  if (urlPath === "/") urlPath = "/index.html";
  if (urlPath === "/pad") urlPath = "/pad.html";

  // prevent path traversal
  const filePath = path.join(ROOT, path.normalize(urlPath));
  if (!filePath.startsWith(ROOT)) { res.writeHead(403); res.end("Forbidden"); return; }

  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404); res.end("Not found"); return; }
    res.writeHead(200, { "Content-Type": MIME[path.extname(filePath)] || "application/octet-stream" });
    res.end(data);
  });
});

// ---- WebSocket relay ----
const wss = new WebSocketServer({ server });
const clients = new Set();

wss.on("connection", (ws) => {
  clients.add(ws);
  ws.role = "unknown";
  ws.on("message", (raw) => {
    let msg;
    try { msg = JSON.parse(raw.toString()); } catch { return; }
    if (msg.type === "hello") { ws.role = msg.role; return; }
    // relay pad -> everyone else (the display)
    for (const c of clients) {
      if (c !== ws && c.readyState === c.OPEN) {
        try { c.send(JSON.stringify(msg)); } catch {}
      }
    }
  });
  ws.on("close", () => clients.delete(ws));
});

function lanIPs() {
  const out = [];
  const ifs = os.networkInterfaces();
  for (const name of Object.keys(ifs)) {
    for (const ni of ifs[name]) {
      if (ni.family === "IPv4" && !ni.internal) out.push(ni.address);
    }
  }
  return out;
}

server.listen(PORT, "0.0.0.0", () => {
  const ips = lanIPs();
  console.log("\n  CNN 3D 시각화 서버가 실행되었습니다.\n");
  console.log(`  노트북 화면 : http://localhost:${PORT}/`);
  for (const ip of ips) {
    console.log(`  휴대폰 패드 : http://${ip}:${PORT}/pad   (같은 와이파이에서 접속)`);
  }
  console.log("\n  종료하려면 Ctrl+C\n");
});
