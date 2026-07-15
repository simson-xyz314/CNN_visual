import puppeteer from "puppeteer-core";
import fs from "fs";

const OUT = "C:/Users/HP/AppData/Local/Temp/claude/c--Users-HP-Desktop-WorkSpace-remnant-CNN-visualize/79053d05-752b-4148-ba30-425af9298e7e/scratchpad/shots";
fs.mkdirSync(OUT, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: "new",
  args: ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist", "--enable-webgl", "--window-size=1600,900"],
  defaultViewport: { width: 1600, height: 900 },
});
const page = await browser.newPage();
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push("CONSOLE " + m.text()); });
page.on("pageerror", (e) => errors.push("PAGEERR " + e.message));

await page.goto("http://localhost:8123/index.html", { waitUntil: "networkidle2" });
await page.waitForFunction("window.__viz && window.__viz.ready", { timeout: 15000 });

// draw a real "3" glyph, downscale to 28x28, run inference
await page.evaluate(() => {
  const c = document.createElement("canvas"); c.width = 280; c.height = 280;
  const x = c.getContext("2d");
  x.fillStyle = "#000"; x.fillRect(0, 0, 280, 280);
  x.fillStyle = "#fff"; x.font = "bold 210px sans-serif"; x.textAlign = "center"; x.textBaseline = "middle";
  x.fillText("3", 140, 140);
  const t = document.createElement("canvas"); t.width = 28; t.height = 28;
  const tx = t.getContext("2d"); tx.fillStyle = "#000"; tx.fillRect(0, 0, 28, 28);
  tx.drawImage(c, 30, 20, 220, 240, 4, 4, 20, 20);
  const d = tx.getImageData(0, 0, 28, 28).data;
  const arr = new Array(784);
  for (let i = 0; i < 784; i++) arr[i] = d[i * 4] / 255;
  window.__viz.runInput(arr);
});
await sleep(400);

const labels = ["1-input", "2-rgb", "3-kernel", "4-scan", "5-maps", "6-relu",
  "7-pool1", "8-conv2", "9-pool2", "10-flatten", "11-fc", "12-softmax", "13-output"];
for (let step = 1; step <= 13; step++) {
  await page.evaluate((s) => window.__viz.jump(s), step);
  await sleep(step === 4 ? 3500 : 1600); // let reveal animations settle
  await page.screenshot({ path: `${OUT}/step${labels[step - 1]}.png` });
  console.log("shot step", step);
}
// mid-scan capture
await page.evaluate(() => window.__viz.jumpFrac(4, 0.45));
await sleep(1400);
await page.screenshot({ path: `${OUT}/step4b-scan-mid.png` });

const pred = await page.evaluate(() => window.__viz && document.getElementById("result-digit").textContent);
console.log("PREDICTION:", pred);
console.log("ERRORS:", errors.length ? errors.slice(0, 10) : "none");
await browser.close();
