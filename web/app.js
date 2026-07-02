'use strict';

const TOKEN_KEY = 'mh_token';
let token = localStorage.getItem(TOKEN_KEY);
let conversationId = null;
let intent = 'ask';            // 'ask' | 'record'
let stagedFiles = [];          // [{ file, type }]
let busy = false;
let recording = false;         // voice-note recording in progress
let mediaRecorder = null;
let audioChunks = [];

const $ = (id) => document.getElementById(id);
const pinScreen = $('pin-screen');
const chatScreen = $('chat-screen');
const pinInput = $('pin-input');
const pinBtn = $('pin-btn');
const pinError = $('pin-error');
const logoutBtn = $('logout-btn');
const healthBanner = $('health-banner');
const messages = $('messages');
const attachBtn = $('attach-btn');
const fileInput = $('file-input');
const fileList = $('file-list');
const textInput = $('text-input');
const sendBtn = $('send-btn');
const intentToggle = $('intent-toggle');
const micBtn = $('mic-btn');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const scrollDown = () => { messages.scrollTop = messages.scrollHeight; };

// --- screens ---------------------------------------------------------------

function showChat() {
  pinScreen.hidden = true;
  chatScreen.hidden = false;
  checkHealth();
  updateReminderBadge();
}

function showPin() {
  chatScreen.hidden = true;
  pinScreen.hidden = false;
  pinInput.focus();
}

function logout() {
  token = null;
  conversationId = null;
  localStorage.removeItem(TOKEN_KEY);
  showPin();
}

// --- networking ------------------------------------------------------------

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers, { Authorization: 'Bearer ' + token });
  const res = await fetch(path, Object.assign({}, opts, { headers }));
  if (res.status === 401) {
    logout();
    throw new Error('Session expired — please unlock again.');
  }
  return res;
}

async function errText(res) {
  const body = await res.json().catch(() => ({}));
  return body.detail || ('HTTP ' + res.status);
}

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    healthBanner.hidden = !!data.ollama_reachable;
  } catch {
    healthBanner.hidden = false;
  }
}

// --- login -----------------------------------------------------------------

async function doLogin() {
  const pin = pinInput.value.trim();
  if (!pin) return;
  pinError.hidden = true;
  try {
    const res = await fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pin }),
    });
    if (!res.ok) throw new Error(await errText(res));
    token = (await res.json()).token;
    localStorage.setItem(TOKEN_KEY, token);
    pinInput.value = '';
    showChat();
  } catch (e) {
    pinError.textContent = e.message;
    pinError.hidden = false;
  }
}

// --- message bubbles -------------------------------------------------------

function addMessage(roleClass, text) {
  const el = document.createElement('div');
  el.className = 'msg ' + roleClass;
  const textSpan = document.createElement('span');
  textSpan.className = 'text';
  textSpan.textContent = text || '';
  el.appendChild(textSpan);
  messages.appendChild(el);
  scrollDown();
  return { el, textSpan };
}

function setSources(el, sources) {
  if (!sources || !sources.length) return;
  const div = document.createElement('div');
  div.className = 'sources';
  div.appendChild(document.createTextNode('Sources: '));
  sources.forEach((s, i) => {
    if (i > 0) div.appendChild(document.createTextNode(', '));
    if (s.has_file) {
      const link = document.createElement('a');
      link.className = 'source-link';
      link.textContent = s.filename;
      link.href = '#';
      link.onclick = (e) => {
        e.preventDefault();
        downloadDocument(s.doc_id, s.filename);
      };
      div.appendChild(link);
    } else {
      div.appendChild(document.createTextNode(s.filename));
    }
  });
  el.appendChild(div);
  scrollDown();
}

async function downloadDocument(docId, filename) {
  try {
    const res = await api('/api/documents/' + encodeURIComponent(docId) + '/file');
    if (!res.ok) throw new Error(await errText(res));
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert('Could not download "' + filename + '": ' + e.message);
  }
}

function setBusy(b) {
  busy = b;
  sendBtn.disabled = b;
}

// --- send dispatch ---------------------------------------------------------

async function onSend() {
  if (busy || recording) return;
  if (stagedFiles.length) {
    await sendFiles();
    return;
  }
  const text = textInput.value.trim();
  if (!text) return;
  textInput.value = '';
  autoGrow();
  if (intent === 'record') {
    await sendHistory(text);
  } else {
    await sendChat(text);
  }
}

async function sendHistory(text) {
  setBusy(true);
  addMessage('user', text);
  const status = addMessage('bot status', 'Saving…');
  try {
    const res = await api('/api/history', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error(await errText(res));
    const data = await res.json().catch(() => ({}));
    status.el.className = 'msg bot';
    status.textSpan.textContent = 'Saved to your record.';
    const drafted = (data.reminders || []).length;
    if (drafted) {
      updateReminderBadge();
      const note = addMessage('bot reminder-note', '');
      note.textSpan.textContent =
        `🔔 ${drafted} follow-up reminder${drafted > 1 ? 's' : ''} detected — ` +
        'tap the bell to review and add to your calendar.';
    }
  } catch (e) {
    status.textSpan.textContent = 'Could not save: ' + e.message;
  } finally {
    setBusy(false);
  }
}

const STATUS_LABELS = {
  routing: 'Understanding your question…',
  searching: 'Searching your records…',
  retrying: 'Refining the search…',
  answering: 'Writing the answer…',
};

async function sendChat(text) {
  setBusy(true);
  addMessage('user', text);
  const bubble = addMessage('bot', '');
  let answer = '';
  let drafted = false;
  try {
    const res = await api('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, conversation_id: conversationId }),
    });
    if (!res.ok) throw new Error(await errText(res));

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const evt = parseSSE(buf.slice(0, idx));
        buf = buf.slice(idx + 2);
        if (!evt) continue;
        if (evt.event === 'status') {
          // Agent progress (routing -> searching -> retrying -> answering).
          // Shown as a placeholder until the first answer token replaces it.
          if (!answer) {
            bubble.el.className = 'msg bot status';
            bubble.textSpan.textContent = STATUS_LABELS[evt.data.step] || 'Working…';
            scrollDown();
          }
        } else if (evt.event === 'token') {
          answer += evt.data.text || '';
          bubble.el.className = 'msg bot';
          // While streaming, show plain text with chart JSON hidden behind a
          // placeholder; the real chart + Markdown render once the reply ends.
          bubble.textSpan.textContent = stripChartsForStreaming(answer);
          scrollDown();
        } else if (evt.event === 'sources') {
          setSources(bubble.el, evt.data.sources);
        } else if (evt.event === 'proposal') {
          // A scheduling turn drafted a reminder — flag it for review.
          drafted = true;
          updateReminderBadge();
        } else if (evt.event === 'done') {
          conversationId = evt.data.conversation_id;
        } else if (evt.event === 'error') {
          throw new Error(evt.data.message || 'Chat error');
        }
      }
    }
    if (answer) renderRich(bubble.textSpan, answer);
    else bubble.textSpan.textContent = '(no response)';
    if (drafted) {
      const note = addMessage('bot reminder-note', '');
      note.textSpan.textContent =
        '🔔 Reminder drafted — tap the bell to review and add it to your calendar.';
    }
    scrollDown();
  } catch (e) {
    bubble.el.className = 'msg bot status';
    bubble.textSpan.textContent = 'Error: ' + e.message;
  } finally {
    setBusy(false);
  }
}

function parseSSE(block) {
  let event = 'message';
  let data = '';
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) data += line.slice(5).trim();
  }
  if (!data) return null;
  try {
    return { event, data: JSON.parse(data) };
  } catch {
    return null;
  }
}

async function sendFiles() {
  const batch = stagedFiles.slice();
  stagedFiles = [];
  renderFileList();

  setBusy(true);
  for (const entry of batch) {
    addMessage('user', '📎 ' + entry.file.name);
    const status = addMessage('bot status', 'Uploading…');
    try {
      const form = new FormData();
      form.append('file', entry.file);
      form.append('type', entry.type);
      const res = await api('/api/documents', { method: 'POST', body: form });
      if (!res.ok) throw new Error(await errText(res));
      const jobId = (await res.json()).job_id;
      pollJob(jobId, status);  // continues in the background
    } catch (e) {
      status.textSpan.textContent = 'Upload failed: ' + e.message;
    }
  }
  setBusy(false);
}

const MAX_POLL_FAILURES = 5;  // tolerate brief network blips before giving up

async function pollJob(jobId, status) {
  let failures = 0;
  while (true) {
    await sleep(2000);
    let job;
    try {
      const res = await api('/api/jobs/' + encodeURIComponent(jobId));
      if (!res.ok) throw new Error(await errText(res));
      job = await res.json();
      failures = 0;
    } catch (e) {
      failures += 1;
      if (failures >= MAX_POLL_FAILURES) {
        status.textSpan.textContent =
          'Lost track of the upload (' + e.message + '). It may still be '
          + 'processing on the server — check back in the document list later.';
        return;
      }
      status.textSpan.textContent =
        'Reconnecting… (' + failures + '/' + MAX_POLL_FAILURES + ')';
      continue;
    }
    const last = (job.progress && job.progress.length)
      ? job.progress[job.progress.length - 1]
      : job.status;
    status.textSpan.textContent = last;
    if (job.status === 'done') {
      status.el.className = 'msg bot';
      return;
    }
    if (job.status === 'error') {
      status.textSpan.textContent = 'Failed: ' + (job.error || 'unknown error');
      return;
    }
  }
}

// --- rich rendering: Markdown + inline SVG charts --------------------------
// The assistant streams Markdown that may embed ```chart fenced JSON blocks.
// During streaming we show plain text (charts hidden); when the reply finishes
// we render Markdown to DOM and turn each chart block into a self-contained
// SVG. Everything is built with createTextNode / DOM APIs (no innerHTML), so
// model output is never interpreted as HTML.

const CHART_FENCE_RE = /```chart\s*([\s\S]*?)```/g;

// Replace chart JSON with a short placeholder mid-stream so the raw JSON never
// flashes in the bubble; handles a not-yet-closed trailing fence too.
function stripChartsForStreaming(text) {
  let out = text.replace(CHART_FENCE_RE, '\n📊 building chart…\n');
  const open = out.lastIndexOf('```chart');
  if (open !== -1) out = out.slice(0, open).trimEnd() + '\n📊 building chart…\n';
  return out;
}

function renderRich(container, fullText) {
  container.classList.add('rich');
  container.textContent = '';
  CHART_FENCE_RE.lastIndex = 0;
  let last = 0;
  let m;
  while ((m = CHART_FENCE_RE.exec(fullText)) !== null) {
    if (m.index > last) appendMarkdown(container, fullText.slice(last, m.index));
    const chart = tryBuildChart(m[1]);
    if (chart) container.appendChild(chart);
    else appendMarkdown(container, m[0]);   // unparseable -> show as-is
    last = m.index + m[0].length;
  }
  if (last < fullText.length) appendMarkdown(container, fullText.slice(last));
}

// --- minimal Markdown (block level) ---------------------------------------

function appendMarkdown(parent, md) {
  const lines = md.replace(/\r/g, '').split('\n');
  let i = 0;
  let para = [];

  const flushPara = () => {
    if (!para.length) return;
    const p = document.createElement('p');
    appendInline(p, para.join(' '));
    parent.appendChild(p);
    para = [];
  };

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) { flushPara(); i++; continue; }

    // Heading
    const h = trimmed.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      flushPara();
      const el = document.createElement('h' + Math.min(h[1].length + 2, 6));
      appendInline(el, h[2]);
      parent.appendChild(el);
      i++; continue;
    }

    // GFM pipe table: a header row followed by a |---|---| separator
    if (trimmed.includes('|') && i + 1 < lines.length
        && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1])
        && lines[i + 1].includes('-')) {
      flushPara();
      const rows = [];
      while (i < lines.length && lines[i].trim().includes('|')) {
        rows.push(lines[i]); i++;
      }
      parent.appendChild(buildTable(rows));
      continue;
    }

    // List (bullets or numbers)
    if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
      flushPara();
      const ordered = /^\s*\d+[.)]\s+/.test(line);
      const list = document.createElement(ordered ? 'ol' : 'ul');
      while (i < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) {
        const li = document.createElement('li');
        appendInline(li, lines[i].replace(/^\s*([-*+]|\d+[.)])\s+/, ''));
        list.appendChild(li);
        i++;
      }
      parent.appendChild(list);
      continue;
    }

    para.push(trimmed);
    i++;
  }
  flushPara();
}

// Inline: **bold**, *italic*, `code`. Builds text/elem nodes (auto-escaped).
const INLINE_RE = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;

function appendInline(parent, text) {
  let last = 0;
  let m;
  INLINE_RE.lastIndex = 0;
  while ((m = INLINE_RE.exec(text)) !== null) {
    if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
    const tok = m[0];
    let el;
    if (tok.startsWith('**')) { el = document.createElement('strong'); el.textContent = tok.slice(2, -2); }
    else if (tok.startsWith('`')) { el = document.createElement('code'); el.textContent = tok.slice(1, -1); }
    else { el = document.createElement('em'); el.textContent = tok.slice(1, -1); }
    parent.appendChild(el);
    last = m.index + tok.length;
  }
  if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
}

function splitRow(line) {
  return line.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
}

function buildTable(rows) {
  const headers = splitRow(rows[0]);
  // Locate a "result/value" column and a "reference/range" column so out-of-
  // range readings can be flagged red, the same way the chart does.
  const valueCol = headers.findIndex((h) => /result|value|reading|level/i.test(h));
  const refCol = headers.findIndex((h) => /referen|range|normal/i.test(h));

  const table = document.createElement('table');
  table.className = 'md-table';
  const thead = document.createElement('thead');
  const htr = document.createElement('tr');
  headers.forEach((h) => { const th = document.createElement('th'); appendInline(th, h); htr.appendChild(th); });
  thead.appendChild(htr);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  for (let r = 2; r < rows.length; r++) {
    const cells = splitRow(rows[r]);
    if (!cells.length) continue;
    const tr = document.createElement('tr');
    const range = refCol >= 0 ? parseRange(cells[refCol]) : null;
    cells.forEach((c, ci) => {
      const td = document.createElement('td');
      appendInline(td, c);
      if (ci === valueCol && range && isOutOfRange(parseNumber(c), range)) {
        td.classList.add('oor');
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  const wrap = document.createElement('div');
  wrap.className = 'md-table-wrap';
  wrap.appendChild(table);
  return wrap;
}

// --- range parsing (shared by tables and charts) ---------------------------

function parseNumber(s) {
  const m = String(s).match(/-?\d+(?:\.\d+)?/);
  return m ? parseFloat(m[0]) : NaN;
}

// Parse "13.0 - 17.0", "<6.0", ">= 40", "0–5" → {low, high} (either may be null).
function parseRange(s) {
  if (s == null) return null;
  const t = String(s).replace(/–|—/g, '-');
  let m = t.match(/(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)/);
  if (m) return { low: parseFloat(m[1]), high: parseFloat(m[2]) };
  m = t.match(/(<=?|>=?)\s*(-?\d+(?:\.\d+)?)/);
  if (m) {
    const v = parseFloat(m[2]);
    return m[1][0] === '<' ? { low: null, high: v } : { low: v, high: null };
  }
  return null;
}

function isOutOfRange(v, range) {
  if (!Number.isFinite(v) || !range) return false;
  return (range.low != null && v < range.low) || (range.high != null && v > range.high);
}

// --- SVG chart -------------------------------------------------------------

const SVGNS = 'http://www.w3.org/2000/svg';

function svg(tag, attrs, text) {
  const el = document.createElementNS(SVGNS, tag);
  for (const k in attrs) el.setAttribute(k, String(attrs[k]));
  if (text != null) el.textContent = text;
  return el;
}

function fmtNum(v) {
  return Number.isInteger(v) ? String(v) : (Math.round(v * 100) / 100).toString();
}

function fmtLabel(s) {
  const m = String(s).match(/^(\d{4})-(\d{2})-(\d{2})$/);
  return m ? `${m[2]}/${m[3]}` : String(s);
}

function tryBuildChart(raw) {
  let spec;
  try { spec = JSON.parse(raw.trim()); } catch { return null; }
  const rawSeries = Array.isArray(spec.series) ? spec.series
    : Array.isArray(spec.data) ? spec.data : null;
  if (!rawSeries) return null;
  const points = rawSeries
    .map((p) => ({
      label: String(p.date ?? p.label ?? p.x ?? ''),
      value: Number(p.value ?? p.y),
    }))
    .filter((p) => Number.isFinite(p.value));
  if (!points.length) return null;
  try { return buildChartFigure(spec, points); } catch { return null; }
}

function buildChartFigure(spec, points) {
  const W = 360, H = 250, mL = 46, mR = 16, mT = 34, mB = 46;
  const x0 = mL, x1 = W - mR, y0 = mT, y1 = H - mB;
  const isBar = String(spec.type).toLowerCase() === 'bar';

  const ref = spec.reference || spec.range || {};
  const lo = Number.isFinite(Number(ref.low)) ? Number(ref.low) : null;
  const hi = Number.isFinite(Number(ref.high)) ? Number(ref.high) : null;
  const range = (lo != null || hi != null) ? { low: lo, high: hi } : null;

  const vals = points.map((p) => p.value);
  let dMin = Math.min(...vals, ...(lo != null ? [lo] : []), ...(isBar ? [0] : []));
  let dMax = Math.max(...vals, ...(hi != null ? [hi] : []));
  if (dMin === dMax) { dMin -= 1; dMax += 1; }
  const pad = (dMax - dMin) * 0.12;
  dMin -= pad; dMax += pad;

  const sy = (v) => y1 - ((v - dMin) / (dMax - dMin)) * (y1 - y0);
  const n = points.length;
  const sx = isBar
    ? (i) => x0 + (i + 0.5) * (x1 - x0) / n
    : (i) => (n === 1 ? (x0 + x1) / 2 : x0 + (i / (n - 1)) * (x1 - x0));
  const oor = (v) => isOutOfRange(v, range);

  const root = svg('svg', {
    viewBox: `0 0 ${W} ${H}`, width: '100%', class: 'chart',
    role: 'img', 'aria-label': spec.title || 'chart',
  });

  // Title
  if (spec.title) root.appendChild(svg('text', { x: x0, y: 18, class: 'ct-title' }, spec.title));

  // Reference band / bounds
  if (range) {
    const top = sy(hi != null ? hi : dMax);
    const bot = sy(lo != null ? lo : dMin);
    root.appendChild(svg('rect', {
      x: x0, y: Math.min(top, bot), width: x1 - x0,
      height: Math.abs(bot - top), class: 'ct-band',
    }));
  }

  // Y gridlines + labels (min, mid, max)
  [dMin + pad, (dMin + dMax) / 2, dMax - pad].forEach((v) => {
    const y = sy(v);
    root.appendChild(svg('line', { x1: x0, y1: y, x2: x1, y2: y, class: 'ct-grid' }));
    root.appendChild(svg('text', { x: x0 - 6, y: y + 3, class: 'ct-axis ct-axis-y' }, fmtNum(v)));
  });

  // X baseline
  root.appendChild(svg('line', { x1: x0, y1: y1, x2: x1, y2: y1, class: 'ct-axis-line' }));

  // Series
  if (isBar) {
    const bw = Math.max(6, ((x1 - x0) / n) * 0.6);
    points.forEach((p, i) => {
      const x = sx(i), y = sy(p.value);
      root.appendChild(svg('rect', {
        x: x - bw / 2, y, width: bw, height: y1 - y,
        class: 'ct-bar' + (oor(p.value) ? ' oor' : ''),
      }));
    });
  } else {
    const d = points.map((p, i) => `${sx(i)},${sy(p.value)}`).join(' ');
    root.appendChild(svg('polyline', { points: d, class: 'ct-line' }));
  }

  // Point markers + value labels
  points.forEach((p, i) => {
    const x = sx(i), y = sy(p.value);
    const bad = oor(p.value);
    if (!isBar) {
      root.appendChild(svg('circle', { cx: x, cy: y, r: 3.5, class: 'ct-dot' + (bad ? ' oor' : '') }));
    }
    if (n <= 7) {
      root.appendChild(svg('text', {
        x, y: y - 7, class: 'ct-val' + (bad ? ' oor' : ''),
      }, fmtNum(p.value)));
    }
    // X label (thin out when crowded)
    if (n <= 7 || i === 0 || i === n - 1 || i % Math.ceil(n / 6) === 0) {
      root.appendChild(svg('text', { x, y: y1 + 14, class: 'ct-axis' }, fmtLabel(p.label)));
    }
  });

  const fig = document.createElement('figure');
  fig.className = 'chart-fig';
  fig.appendChild(root);
  const cap = ref.label || (range
    ? `Reference: ${lo != null ? fmtNum(lo) : ''}${lo != null && hi != null ? '–' : ''}${hi != null ? fmtNum(hi) : ''}`
    : '');
  if (spec.unit || cap) {
    const fc = document.createElement('figcaption');
    fc.textContent = [spec.unit ? `Unit: ${spec.unit}` : '', cap].filter(Boolean).join('  ·  ');
    fig.appendChild(fc);
  }
  return fig;
}

// --- reminders -------------------------------------------------------------

function browserTz() {
  try { return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'; }
  catch { return 'UTC'; }
}

function rruleText(rrule) {
  if (!rrule) return 'One-off';
  const freq = (/FREQ=([A-Z]+)/.exec(rrule) || [, 'recurring'])[1].toLowerCase();
  const count = /COUNT=(\d+)/.exec(rrule);
  return count ? `Repeats ${freq} ×${count[1]}` : `Repeats ${freq}`;
}

async function updateReminderBadge() {
  const badge = $('reminders-badge');
  if (!badge) return;
  try {
    const res = await api('/api/reminders?status=pending');
    if (!res.ok) return;
    const n = ((await res.json()).reminders || []).length;
    badge.textContent = n;
    badge.hidden = n === 0;
  } catch { /* ignore badge errors */ }
}

async function loadGoogleStatus() {
  const box = $('google-status');
  if (!box) return;
  box.innerHTML = '';
  let data;
  try { data = await (await api('/api/google/status')).json(); }
  catch { return; }
  if (data.authorized) {
    box.innerHTML = '<span class="gs-ok">✓ Google Calendar connected</span>';
    return;
  }
  if (!data.credentials_present) {
    box.innerHTML =
      'Google not set up — add <code>rag_db/google_credentials.json</code> ' +
      'to enable calendar reminders. You can still review and dismiss drafts.';
    return;
  }
  box.innerHTML = 'Google Calendar not connected. ';
  const btn = document.createElement('button');
  btn.className = 'mini-btn';
  btn.textContent = 'Connect';
  btn.onclick = async () => {
    btn.disabled = true;
    btn.textContent = 'Approve in the browser on the server…';
    try {
      const res = await api('/api/google/authorize', { method: 'POST' });
      if (!res.ok) throw new Error(await errText(res));
      loadGoogleStatus();
    } catch (e) {
      alert('Authorize failed: ' + e.message);
      btn.disabled = false;
      btn.textContent = 'Connect';
    }
  };
  box.appendChild(btn);
}

async function loadReminders() {
  const list = $('reminders-list');
  if (!list) return;
  list.textContent = 'Loading…';
  let items;
  try { items = ((await (await api('/api/reminders?status=pending')).json()).reminders) || []; }
  catch { list.textContent = 'Could not load reminders.'; return; }
  list.innerHTML = '';
  if (!items.length) {
    list.innerHTML = '<p class="muted">Nothing to review right now.</p>';
    return;
  }
  items.forEach((r) => list.appendChild(reminderCard(r)));
}

function reminderCard(r) {
  const card = document.createElement('div');
  card.className = 'reminder-card';

  const kind = document.createElement('span');
  kind.className = 'reminder-kind ' + r.kind;
  kind.textContent = r.kind;

  const text = document.createElement('input');
  text.className = 'reminder-text';
  text.value = r.proposed_text || r.title || '';

  const when = document.createElement('input');
  when.type = 'datetime-local';
  when.className = 'reminder-when';
  when.value = (r.start || '').slice(0, 16);   // ISO -> datetime-local

  const rep = document.createElement('span');
  rep.className = 'reminder-rep';
  rep.textContent = rruleText(r.rrule);

  const actions = document.createElement('div');
  actions.className = 'reminder-actions';

  const removeIfEmpty = () => {
    card.remove();
    updateReminderBadge();
    if (!$('reminders-list').children.length) loadReminders();
  };

  const approve = document.createElement('button');
  approve.className = 'mini-btn primary';
  approve.textContent = 'Approve';
  approve.onclick = async () => {
    if (!when.value) { alert('Set a date and time first.'); return; }
    approve.disabled = true;
    approve.textContent = 'Adding…';
    const body = {
      title: text.value.trim(),
      proposed_text: text.value.trim(),
      start: when.value.length === 16 ? when.value + ':00' : when.value,
      timezone: browserTz(),
    };
    try {
      const res = await api(`/api/reminders/${r.id}/confirm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await errText(res));
      removeIfEmpty();
    } catch (e) {
      approve.disabled = false;
      approve.textContent = 'Approve';
      alert('Could not add to calendar: ' + e.message);
    }
  };

  const dismiss = document.createElement('button');
  dismiss.className = 'mini-btn';
  dismiss.textContent = 'Dismiss';
  dismiss.onclick = async () => {
    try {
      await api(`/api/reminders/${r.id}/dismiss`, { method: 'POST' });
      removeIfEmpty();
    } catch (e) { alert(e.message); }
  };

  actions.append(approve, dismiss);
  card.append(kind, text, when, rep, actions);
  return card;
}

function openReminders() {
  const panel = $('reminders-panel');
  if (!panel) return;
  panel.hidden = false;
  loadGoogleStatus();
  loadReminders();
}

function closeReminders() {
  const panel = $('reminders-panel');
  if (panel) panel.hidden = true;
}

// --- composer controls -----------------------------------------------------

function autoGrow() {
  textInput.style.height = 'auto';
  textInput.style.height = Math.min(textInput.scrollHeight, 120) + 'px';
}

function updatePlaceholder() {
  textInput.placeholder = intent === 'record'
    ? 'Share medical history or current status…'
    : 'Ask a question…';
}

function renderFileList() {
  fileList.innerHTML = '';
  if (!stagedFiles.length) {
    fileList.hidden = true;
    textInput.disabled = false;
    intentToggle.classList.remove('disabled');
    updatePlaceholder();
    return;
  }
  fileList.hidden = false;
  textInput.disabled = true;
  textInput.placeholder = 'Sending file(s)…';
  intentToggle.classList.add('disabled');

  stagedFiles.forEach((entry, i) => {
    const row = document.createElement('div');
    row.className = 'file-row';

    const name = document.createElement('span');
    name.className = 'file-name';
    name.textContent = entry.file.name;

    const sel = document.createElement('select');
    sel.className = 'file-type';
    sel.innerHTML =
      '<option value="report">Report</option>' +
      '<option value="prescription">Prescription</option>';
    sel.value = entry.type;
    sel.onchange = () => { stagedFiles[i].type = sel.value; };

    const remove = document.createElement('button');
    remove.className = 'link-btn';
    remove.title = 'Remove';
    remove.textContent = '✕';
    remove.onclick = () => { stagedFiles.splice(i, 1); renderFileList(); };

    row.append(name, sel, remove);
    fileList.appendChild(row);
  });
}

attachBtn.onclick = () => fileInput.click();

fileInput.onchange = () => {
  for (const f of fileInput.files) {
    stagedFiles.push({ file: f, type: 'report' });
  }
  fileInput.value = '';
  renderFileList();
};

function setIntent(name) {
  intent = name;
  intentToggle.querySelectorAll('button').forEach((b) =>
    b.classList.toggle('active', b.dataset.intent === name));
  updatePlaceholder();
}

intentToggle.querySelectorAll('button').forEach((btn) => {
  btn.onclick = () => setIntent(btn.dataset.intent);
});

// --- voice input -----------------------------------------------------------

async function toggleMic() {
  if (recording) {
    stopRecording();
  } else if (!busy) {
    await startRecording();
  }
}

async function startRecording() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    alert('Microphone unavailable: ' + e.message);
    return;
  }
  audioChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size) audioChunks.push(e.data);
  };
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    const type = mediaRecorder.mimeType || 'audio/webm';
    transcribeBlob(new Blob(audioChunks, { type }));
  };
  mediaRecorder.start();
  recording = true;
  micBtn.classList.add('recording');
  micBtn.title = 'Stop recording';
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  recording = false;
  micBtn.classList.remove('recording');
  micBtn.title = 'Record voice note';
}

async function transcribeBlob(blob) {
  if (!blob.size) return;
  setBusy(true);
  micBtn.disabled = true;
  textInput.placeholder = 'Transcribing…';
  try {
    const ext = blob.type.includes('ogg') ? 'ogg' : 'webm';
    const form = new FormData();
    form.append('file', blob, 'voice.' + ext);
    const res = await api('/api/transcribe', { method: 'POST', body: form });
    if (!res.ok) throw new Error(await errText(res));
    const text = ((await res.json()).text || '').trim();
    if (!text) {
      alert('Nothing was transcribed — please try recording again.');
      return;
    }
    // Drop the transcript into the composer for review — appended if the box
    // already holds text. Nothing is saved until the user taps send.
    // The current intent (Ask vs Add to record) is preserved so voice works
    // for questions as well as new records.
    const existing = textInput.value.trim();
    textInput.value = existing ? existing + ' ' + text : text;
    autoGrow();
    textInput.focus();
  } catch (e) {
    alert('Could not transcribe: ' + e.message);
  } finally {
    setBusy(false);
    micBtn.disabled = false;
    updatePlaceholder();
  }
}

if (navigator.mediaDevices && window.MediaRecorder) {
  micBtn.onclick = toggleMic;
} else {
  micBtn.hidden = true;   // unsupported browser or insecure (non-HTTPS) context
}

textInput.addEventListener('input', autoGrow);
textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    onSend();
  }
});

pinBtn.onclick = doLogin;
pinInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') doLogin();
});
logoutBtn.onclick = logout;
sendBtn.onclick = onSend;
$('reminders-btn').onclick = openReminders;
$('reminders-close').onclick = closeReminders;

// --- init ------------------------------------------------------------------

if (token) {
  showChat();
} else {
  showPin();
}
