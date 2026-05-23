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
    status.el.className = 'msg bot';
    status.textSpan.textContent = 'Saved to your record.';
  } catch (e) {
    status.textSpan.textContent = 'Could not save: ' + e.message;
  } finally {
    setBusy(false);
  }
}

async function sendChat(text) {
  setBusy(true);
  addMessage('user', text);
  const bubble = addMessage('bot', '');
  let answer = '';
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
        if (evt.event === 'token') {
          answer += evt.data.text || '';
          bubble.textSpan.textContent = answer;
          scrollDown();
        } else if (evt.event === 'sources') {
          setSources(bubble.el, evt.data.sources);
        } else if (evt.event === 'done') {
          conversationId = evt.data.conversation_id;
        } else if (evt.event === 'error') {
          throw new Error(evt.data.message || 'Chat error');
        }
      }
    }
    if (!answer) bubble.textSpan.textContent = '(no response)';
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
    const existing = textInput.value.trim();
    textInput.value = existing ? existing + ' ' + text : text;
    // Voice notes are medical history: switch the toggle so the user only has
    // to review and send. The toggle stays visible — this is not hidden.
    setIntent('record');
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

// --- init ------------------------------------------------------------------

if (token) {
  showChat();
} else {
  showPin();
}
