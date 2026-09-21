#!/usr/bin/env node
// Record the real WebUI with an isolated, explicitly labelled fixture bridge.
// Requires Playwright (NODE_PATH is supported) and Python/Pillow. No native inference.
const { chromium } = require('playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const { spawnSync } = require('node:child_process');
const assert = require('node:assert/strict');

const repo = path.resolve(__dirname, '../..');
const uiRoot = path.join(repo, 'Sources/GenImageApp/Resources/WebUI');
const output = path.resolve(process.argv[2] || path.join(repo, 'images/operation-demo.gif'));
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png' };

async function main() {
  const fixture = JSON.parse(await fs.readFile(path.join(__dirname, 'fixture.json'), 'utf8'));
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'genmedia-gif-'));
  const server = http.createServer(async (req, res) => {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    const file = path.resolve(uiRoot, '.' + (pathname === '/' ? '/index.html' : pathname));
    if (!file.startsWith(uiRoot + path.sep)) { res.writeHead(403).end(); return; }
    try {
      const bytes = await fs.readFile(file);
      res.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' });
      res.end(bytes);
    } catch { res.writeHead(404).end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({
      headless: true,
      ...(process.env.CHROMIUM_EXECUTABLE ? { executablePath: process.env.CHROMIUM_EXECUTABLE } : {}),
    });
    const page = await browser.newPage({ viewport: { width: 1440, height: 964 }, deviceScaleFactor: 1 });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    await page.addInitScript(state => {
      localStorage.clear();
      localStorage.setItem('genimage.locale', 'zh-Hant');
      localStorage.setItem('genimage.theme', 'violet');
      localStorage.setItem('genimage.profileHintSeen', 'true');
      window.webkit = { messageHandlers: { genimage: { postMessage({ id, method, params }) {
        let payload = {};
        const sendState = () => window.GenImageNative.receiveState(structuredClone(state));
        switch (method) {
          case 'bootstrap': break;
          case 'updateRecipe': Object.assign(state.recipe, params); break;
          case 'updateVideoOutputSettings': Object.assign(state.videoOutputSettings, params); break;
          case 'updateMusicOutputSettings': Object.assign(state.musicOutputSettings, params); break;
          case 'applyWorkspaceTabDraft':
            for (const key of ['recipe', 'videoOutputSettings', 'musicOutputSettings']) {
              Object.assign(state[key], params[key] || {});
            }
            break;
          case 'clearStatus': state.statusMessage = null; break;
          case 'createAutomaticFlowWorkspace': {
            const workspaceID = '00000000-0000-4000-8000-000000000002';
            state.workspaces.push({ id: workspaceID, name: '簡單 MV · 操作示範', isDefault: false });
            state.selectedWorkspaceID = workspaceID;
            state.projectName = '簡單 MV · 操作示範';
            state.assets = [];
            state.selectedAssetID = null;
            payload = { workspaceID };
            window.GenImageNative.receive({ kind: 'response', id, ok: true, payload });
            setTimeout(sendState, 30);
            return;
          }
          case 'selectAsset': state.selectedAssetID = params.assetID; break;
          default:
            window.GenImageNative.receive({ kind: 'response', id, ok: false, error: `Demo excludes native command: ${method}` });
            throw new Error(`Unexpected native command: ${method}`);
        }
        window.GenImageNative.receive({ kind: 'response', id, ok: true, payload });
        sendState();
      } } } };
    }, fixture);
    await page.goto(`http://127.0.0.1:${server.address().port}/`, { waitUntil: 'networkidle' });
    await page.locator('[data-ui-field="generationType"]').waitFor();
    await page.addStyleTag({ content: `
      #app { height: calc(100vh - 64px); margin-top: 64px; }
      #demo-header { position: fixed; inset: 0 0 auto; height: 64px; z-index: 10000;
        display: flex; align-items: center; gap: 16px; padding: 0 24px;
        background: #171522; color: #f4f0ff; font: 600 20px -apple-system, sans-serif; }
      #demo-step { background: #b8a0ff; color: #211435; padding: 5px 10px; border-radius: 7px; font-size: 15px; }
      #demo-note { margin-left: auto; font-size: 13px; font-weight: 400; color: #c6bfda; }
      #demo-pointer { position: fixed; z-index: 10001; pointer-events: none; width: 23px; height: 30px;
        filter: drop-shadow(0 1px 2px #0008); }
      #demo-pointer.pulse::before { content: ''; position: absolute; width: 32px; height: 32px;
        border: 3px solid #b78aff; border-radius: 50%; left: -14px; top: -14px; background: #b78aff33; }
    ` });
    await page.evaluate(() => {
      const header = document.createElement('div'); header.id = 'demo-header';
      header.innerHTML = '<span id="demo-step">01 / 05</span><span id="demo-title"></span><span id="demo-note">GenMedia · 介面操作示範 / UI demo · 示範資料</span>';
      const pointer = document.createElement('div'); pointer.id = 'demo-pointer';
      pointer.innerHTML = '<svg viewBox="0 0 24 30"><path d="M2 2v23l6-6 5 9 4-2-5-9h9z" fill="white" stroke="#241c33" stroke-width="1.5"/></svg>';
      document.body.append(header, pointer);
    });

    const frames = [];
    let point = { x: 650, y: 480 };
    async function shot(duration) {
      const name = `frame-${String(frames.length).padStart(4, '0')}.png`;
      await page.screenshot({ path: path.join(temp, name) });
      frames.push({ name, duration });
    }
    async function stage(number, title) {
      await page.evaluate(({ number, title }) => {
        document.querySelector('#demo-step').textContent = `${String(number).padStart(2, '0')} / 05`;
        document.querySelector('#demo-title').textContent = title;
      }, { number, title });
    }
    async function move(selector) {
      const element = page.locator(selector).first();
      await element.waitFor({ state: 'visible' });
      const box = await element.boundingBox();
      const target = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
      const from = { ...point };
      for (let i = 1; i <= 5; i++) {
        const t = i / 5;
        point = { x: from.x + (target.x - from.x) * t, y: from.y + (target.y - from.y) * t };
        await page.mouse.move(point.x, point.y);
        await page.evaluate(({ x, y }) => {
          Object.assign(document.querySelector('#demo-pointer').style, { left: `${x}px`, top: `${y}px` });
        }, point);
        await shot(0.08);
      }
      return element;
    }
    async function click(selector) {
      const element = await move(selector);
      await page.evaluate(() => document.querySelector('#demo-pointer').classList.add('pulse'));
      await shot(0.16);
      await element.click();
      await page.waitForTimeout(180);
      await page.evaluate(() => document.querySelector('#demo-pointer').classList.remove('pulse'));
    }
    async function select(selector, value) {
      const element = await move(selector);
      await element.selectOption(value);
      await page.waitForTimeout(300);
    }
    async function type(selector, value, chunkSize = 5) {
      await click(selector);
      const element = page.locator(selector).first();
      for (let i = chunkSize; i < value.length + chunkSize; i += chunkSize) {
        await element.fill(value.slice(0, i));
        await shot(0.16);
      }
      await element.blur();
      await page.waitForTimeout(500);
    }

    await stage(1, '輸入提示詞，設定圖片尺寸');
    await shot(1.0);
    await type('#recipe-prompt', '晨光中的山間小屋，薄霧、松林，電影感構圖。');
    await click('[data-action="promptTab"][data-tab="imageOutput"]');
    await select('[data-aspect-ratio-select][data-output-kind="image"]', '16:9');
    assert.equal(await page.locator('[data-aspect-ratio-select][data-output-kind="image"]').inputValue(), '16:9');
    await shot(1.5);
    await stage(2, '切換影片，調整輸出設定');
    await select('[data-ui-field="generationType"]', 'video');
    await click('[data-action="promptTab"][data-tab="videoOutput"]');
    await shot(2.2);
    await stage(3, '切換音樂，加入歌詞與風格');
    await select('[data-ui-field="generationType"]', 'music');
    await click('[data-action="promptTab"][data-tab="lyrics"]');
    await type('[data-music-field="lyrics"]', '[Verse]\n晨光穿過森林，微風帶著旋律。', 6);
    await click('[data-action="promptTab"][data-tab="musicOutput"]');
    await select('[data-music-field="style"]', 'cinematic');
    await shot(1.8);
    await stage(4, '在模型中心搜尋需要的模型');
    await click('[data-action="navigate"][data-route="models"]');
    await shot(1.3);
    await type('[data-ui-field="modelSearch"]', 'LTX', 1);
    assert.equal(await page.locator('[data-model-card]').count(), 1);
    assert.match(await page.locator('[data-model-card]').innerText(), /LTX/);
    await shot(1.6);
    await stage(5, '自動流程：主視覺 → 音樂 → 圖片循環 → 影音合併');
    await click('[data-action="navigate"][data-route="automaticFlow"]');
    await shot(2.2);
    await click('[data-action="createAutomaticFlow"][data-template-id="simpleMV"]');
    await page.locator('.workspace-tab-list [role="tab"]').nth(3).waitFor();
    assert.equal(await page.locator('.workspace-tab-list [role="tab"]').count(), 4);
    assert.match(await page.locator('.workspace-tab-list').innerText(), /主視覺.*背景音樂.*圖片循環.*影音合併/s);
    await shot(2.6);
    await fs.copyFile(path.join(temp, frames[frames.length - 1].name), path.join(temp, 'preview.png'));
    if (errors.length) throw new Error(errors.join('\n'));

    await fs.writeFile(path.join(temp, 'frames.json'), JSON.stringify(frames));
    await fs.mkdir(path.dirname(output), { recursive: true });
    const encoded = spawnSync(process.env.PYTHON || 'python3', [path.join(__dirname, 'encode.py'), temp, output], { encoding: 'utf8' });
    if (encoded.status !== 0) throw new Error(encoded.stderr || encoded.error?.message);
    console.log(JSON.stringify({ output, frames: frames.length,
      seconds: frames.reduce((sum, f) => sum + f.duration, 0), bytes: (await fs.stat(output)).size, captures: temp }, null, 2));
  } finally {
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
