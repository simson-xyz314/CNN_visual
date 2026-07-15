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
page.on("pageerror", (e) => errors.push("PAGEERR " + e.message));

await page.goto("http://localhost:8099/index.html", { waitUntil: "networkidle2" });
await page.waitForFunction("window.__viz && window.__viz.ready", { timeout: 15000 });

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
  window.__viz.freeze(true);
});
await sleep(400);

for (const step of [1, 2, 5, 8, 10, 11, 13]) {
  await page.evaluate((s) => window.__viz.jump(s), step);
  await sleep(1500);
  await page.screenshot({ path: `${OUT}/v2-step${step}.png` });
  console.log("shot", step);
}
console.log("ERRORS:", errors.length ? errors.slice(0, 5) : "none");
await browser.close();
