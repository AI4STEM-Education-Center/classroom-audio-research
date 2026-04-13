/**
 * Audio Capture Client — Dual Device
 * ====================================
 * Records from TWO audio devices simultaneously (e.g., USB headset + AUX mic).
 * Each device gets 3 parallel channels:
 *   RAW       — getUserMedia processing OFF → MediaRecorder (webm/opus)
 *   PROCESSED — getUserMedia processing ON  → MediaRecorder (webm/opus)
 *   PCM       — Float32 via ScriptProcessor → direct WAV (no codec)
 *
 * Total: up to 6 parallel recordings per take.
 */

// ── State ───────────────────────────────────────────────────────────

let currentPhase = "setup";
let isRecording = false;
let timerInterval = null;
let timerSeconds = 0;
let meterRAF = null;
let activeAnalysers = [];
let activeCleanups = [];  // functions to call on stop

// ── Logging ─────────────────────────────────────────────────────────

function log(msg, level = "") {
    const el = document.getElementById("log");
    const entry = document.createElement("div");
    entry.className = `entry ${level}`;
    const ts = new Date().toLocaleTimeString();
    entry.textContent = `[${ts}] ${msg}`;
    el.appendChild(entry);
    el.scrollTop = el.scrollHeight;
    console.log(`[${level || "info"}] ${msg}`);
}

// ── Device Enumeration ──────────────────────────────────────────────

async function enumerateDevices() {
    try {
        const tempStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        tempStream.getTracks().forEach(t => t.stop());

        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter(d => d.kind === "audioinput");

        ["deviceSelectA", "deviceSelectB"].forEach(selId => {
            const sel = document.getElementById(selId);
            const prevValue = sel.value;
            sel.innerHTML = '<option value="">(none — disabled)</option>';
            audioInputs.forEach((d, i) => {
                const opt = document.createElement("option");
                opt.value = d.deviceId;
                opt.textContent = d.label || `Microphone ${i + 1}`;
                sel.appendChild(opt);
            });
            // Restore previous selection if still available
            if (prevValue) {
                const exists = Array.from(sel.options).some(o => o.value === prevValue);
                if (exists) sel.value = prevValue;
            }
        });

        log(`Found ${audioInputs.length} audio input device(s)`, "ok");
        if (audioInputs.length === 0) {
            log("No audio inputs found! Check mic connections.", "err");
        }
    } catch (err) {
        log(`Device enumeration failed: ${err.message}`, "err");
    }
}

// ── Stream + Analyser Setup ─────────────────────────────────────────

async function getStream(deviceId, raw = true) {
    const constraints = {
        audio: {
            deviceId: deviceId ? { exact: deviceId } : undefined,
            channelCount: 1,
            sampleRate: 48000,
            echoCancellation: !raw,
            noiseSuppression: !raw,
            autoGainControl: !raw,
        }
    };
    return await navigator.mediaDevices.getUserMedia(constraints);
}

function setupAnalyser(stream, meterId, levelId) {
    const ctx = new AudioContext();
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.3;
    source.connect(analyser);
    return { ctx, analyser, meterId, levelId };
}

function updateMeters() {
    activeAnalysers.forEach(a => {
        if (!a || !a.analyser) return;
        const data = new Float32Array(a.analyser.fftSize);
        a.analyser.getFloatTimeDomainData(data);
        let sum = 0, peak = 0;
        for (let i = 0; i < data.length; i++) {
            sum += data[i] * data[i];
            peak = Math.max(peak, Math.abs(data[i]));
        }
        const rms = Math.sqrt(sum / data.length);
        const rmsDb = rms > 0 ? 20 * Math.log10(rms) : -100;
        const peakDb = peak > 0 ? 20 * Math.log10(peak) : -100;
        const pct = Math.max(0, Math.min(100, ((peakDb + 60) / 60) * 100));
        const meterEl = document.getElementById(a.meterId);
        const levelEl = document.getElementById(a.levelId);
        if (meterEl) meterEl.style.width = pct + "%";
        if (levelEl) levelEl.textContent = rmsDb > -100 ? rmsDb.toFixed(1) : "--";
    });
    meterRAF = requestAnimationFrame(updateMeters);
}

// ── PCM Capture (per device) ────────────────────────────────────────

function createPCMCapture(stream, prefix) {
    const ctx = new AudioContext({ sampleRate: 48000 });
    const source = ctx.createMediaStreamSource(stream);
    const buffers = [];
    const bufSize = 4096;
    const processor = ctx.createScriptProcessor(bufSize, 1, 1);

    // Capture the exact moment the first PCM buffer arrives — this is the true
    // recording start time for the PCM stream, independent of upload latency.
    let pcmStartedAt = null;

    processor.onaudioprocess = (e) => {
        if (!isRecording) return;
        if (!pcmStartedAt) pcmStartedAt = new Date().toISOString();
        buffers.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    source.connect(processor);
    processor.connect(ctx.destination);

    // Analyser for meter
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.3;
    source.connect(analyser);

    return {
        analyserInfo: { ctx, analyser, meterId: `meter${prefix}_pcm`, levelId: `level${prefix}_pcm` },
        getBlob: () => {
            const total = buffers.reduce((a, b) => a + b.length, 0);
            const merged = new Float32Array(total);
            let off = 0;
            for (const b of buffers) { merged.set(b, off); off += b.length; }
            return new Blob([merged.buffer], { type: "application/octet-stream" });
        },
        getStartedAt: () => pcmStartedAt,
        sampleRate: ctx.sampleRate,
        stop: () => {
            processor.disconnect();
            source.disconnect();
            ctx.close();
        },
    };
}

// ── Per-device recording setup ──────────────────────────────────────

async function setupDeviceRecording(deviceId, prefix, tag, usePCM) {
    /**
     * Returns { analysers: [], streams: [], stopAndUpload: async fn }
     * prefix: "A" or "B"
     * tag: user-supplied tag like "usb_headset" or "aux_lapel"
     */
    const analysers = [];
    const streams = [];
    const contexts = [];

    const source = tag || `device_${prefix.toLowerCase()}`;

    // RAW stream + recorder
    const streamRaw = await getStream(deviceId, true);
    streams.push(streamRaw);
    const aRaw = setupAnalyser(streamRaw, `meter${prefix}_raw`, `level${prefix}_raw`);
    analysers.push(aRaw);
    contexts.push(aRaw.ctx);

    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus" : "audio/webm";

    const chunksRaw = [];
    const recRaw = new MediaRecorder(streamRaw, { mimeType });
    recRaw.ondataavailable = (e) => { if (e.data.size > 0) chunksRaw.push(e.data); };

    // PROCESSED stream + recorder
    const streamProc = await getStream(deviceId, false);
    streams.push(streamProc);
    const aProc = setupAnalyser(streamProc, `meter${prefix}_proc`, `level${prefix}_proc`);
    analysers.push(aProc);
    contexts.push(aProc.ctx);

    const chunksProc = [];
    const recProc = new MediaRecorder(streamProc, { mimeType });
    recProc.ondataavailable = (e) => { if (e.data.size > 0) chunksProc.push(e.data); };

    // PCM
    let pcm = null;
    if (usePCM) {
        const pcmStream = await getStream(deviceId, true);
        streams.push(pcmStream);
        pcm = createPCMCapture(pcmStream, prefix);
        analysers.push(pcm.analyserInfo);
        contexts.push(pcm.analyserInfo.ctx);
    }

    log(`[Device ${prefix}] Streams ready: ${source}`, "ok");

    // Capture the client-side wall-clock time at the exact moment recording begins.
    // This is sent to the server so filenames and metadata reflect the true start
    // time rather than the server-receive time (which can be seconds later).
    const recordingStartedAt = new Date().toISOString();
    recRaw.start(1000);
    recProc.start(1000);

    // Return stop + upload function
    const stopAndUpload = async (phase, speaker, deviceLabel, clientName = "unknown") => {
        recRaw.stop();
        recProc.stop();
        await new Promise(r => setTimeout(r, 500));

        streams.forEach(s => s.getTracks().forEach(t => t.stop()));
        contexts.forEach(c => { try { c.close(); } catch {} });

        // Upload RAW
        const blobRaw = new Blob(chunksRaw, { type: mimeType });
        await uploadBlob(blobRaw, {
            phase, source, processing: "raw", speaker, deviceLabel,
            device: prefix, tag, clientName,
            recordingStartedAt,
        });

        // Upload PROCESSED
        const blobProc = new Blob(chunksProc, { type: mimeType });
        await uploadBlob(blobProc, {
            phase, source, processing: "processed", speaker, deviceLabel,
            device: prefix, tag, clientName,
            recordingStartedAt,
        });

        // Upload PCM — prefer the first-buffer timestamp for maximum accuracy;
        // fall back to the MediaRecorder start time if PCM never captured a buffer.
        if (pcm) {
            const pcmBlob = pcm.getBlob();
            const pcmStartedAt = pcm.getStartedAt() || recordingStartedAt;
            pcm.stop();
            await uploadPCM(pcmBlob, {
                phase, source, speaker, deviceLabel,
                sampleRate: pcm.sampleRate,
                device: prefix, tag, clientName,
                recordingStartedAt: pcmStartedAt,
            });
        }
    };

    return { analysers, stopAndUpload };
}

// ── Main Recording ──────────────────────────────────────────────────

async function startRecording() {
    const devA_id = document.getElementById("deviceSelectA").value;
    const devB_id = document.getElementById("deviceSelectB").value;
    const enableA = document.getElementById("chkEnableA").checked && devA_id;
    const enableB = document.getElementById("chkEnableB").checked && devB_id;

    if (!enableA && !enableB) {
        log("Enable at least one device and select a mic!", "err");
        return;
    }

    if (enableA && enableB && devA_id === devB_id) {
        log("WARNING: Both devices are the same mic. Select different devices for A/B comparison.", "warn");
    }

    const speaker = document.getElementById("speakerName").value.trim();
    const clientName = document.getElementById("clientName").value.trim() || "unknown";
    const customPhase = document.getElementById("customPhase").value.trim();
    const phase = customPhase || currentPhase;
    const usePCM = document.getElementById("chkPCM").checked;
    const tagA = document.getElementById("deviceTagA").value.trim() || "device_a";
    const tagB = document.getElementById("deviceTagB").value.trim() || "device_b";

    log(`Starting recording — Phase: ${phase}` +
        (enableA ? ` | Dev A: ${tagA}` : "") +
        (enableB ? ` | Dev B: ${tagB}` : "") +
        ` | Client: ${clientName}`);

    activeAnalysers = [];
    activeCleanups = [];

    try {
        // Setup Device A
        let devASetup = null;
        if (enableA) {
            const labelA = document.getElementById("deviceSelectA").options[
                document.getElementById("deviceSelectA").selectedIndex]?.textContent || "unknown";
            devASetup = await setupDeviceRecording(devA_id, "A", tagA, usePCM);
            activeAnalysers.push(...devASetup.analysers);
            activeCleanups.push(() => devASetup.stopAndUpload(phase, speaker, labelA, clientName));
        }

        // Setup Device B
        let devBSetup = null;
        if (enableB) {
            const labelB = document.getElementById("deviceSelectB").options[
                document.getElementById("deviceSelectB").selectedIndex]?.textContent || "unknown";
            devBSetup = await setupDeviceRecording(devB_id, "B", tagB, usePCM);
            activeAnalysers.push(...devBSetup.analysers);
            activeCleanups.push(() => devBSetup.stopAndUpload(phase, speaker, labelB, clientName));
        }

        // Start meters + timer
        isRecording = true;
        updateMeters();
        startTimer();

        document.getElementById("recIndicator").classList.add("recording");
        document.getElementById("btnRecord").disabled = true;
        document.getElementById("btnStop").disabled = false;

        const totalChannels = (enableA ? (usePCM ? 3 : 2) : 0) + (enableB ? (usePCM ? 3 : 2) : 0);
        log(`Recording on ${totalChannels} channels`, "ok");

        // localStop: stops this client's recording and uploads — no broadcast
        window._localStop = async () => {
            if (!isRecording) return;
            isRecording = false;
            cancelAnimationFrame(meterRAF);
            stopTimer();

            document.getElementById("recIndicator").classList.remove("recording");
            document.getElementById("btnRecord").disabled = false;
            document.getElementById("btnStop").disabled = true;

            log(`Recording stopped (${timerSeconds}s). Uploading...`);

            for (const cleanup of activeCleanups) {
                await cleanup();
            }
            activeAnalysers = [];
            activeCleanups = [];
            log("All uploads complete.", "ok");
        };

        // STOP button: broadcast to all clients, then stop locally
        document.getElementById("btnStop").onclick = async () => {
            await fetch("/api/command/stop", { method: "POST" }).catch(() => {});
            await window._localStop();
        };

    } catch (err) {
        log(`Failed: ${err.message}`, "err");
        isRecording = false;
        // Cleanup any partial setup
        for (const cleanup of activeCleanups) {
            try { await cleanup(); } catch {}
        }
        activeAnalysers = [];
        activeCleanups = [];
    }
}

// ── Upload ──────────────────────────────────────────────────────────

async function uploadBlob(blob, metadata) {
    const formData = new FormData();
    formData.append("file", blob, "recording.webm");
    formData.append("metadata", JSON.stringify(metadata));
    try {
        const resp = await fetch("/api/upload", { method: "POST", body: formData });
        const result = await resp.json();
        log(`Uploaded: ${result.original} (${formatBytes(result.size)}) -> WAV: ${result.wav}`, "ok");
        addRecordingToList(result.original, result.size);
    } catch (err) {
        log(`Upload failed: ${err.message}`, "err");
    }
}

async function uploadPCM(blob, metadata) {
    try {
        const resp = await fetch(`/api/upload-pcm?metadata=${encodeURIComponent(JSON.stringify(metadata))}`, {
            method: "POST",
            headers: { "Content-Type": "application/octet-stream" },
            body: blob,
        });
        const result = await resp.json();
        log(`PCM uploaded: ${result.wav} (${formatBytes(result.size)})`, "ok");
        addRecordingToList(result.wav, result.size);
    } catch (err) {
        log(`PCM upload failed: ${err.message}`, "err");
    }
}

// ── Timer ───────────────────────────────────────────────────────────

function startTimer() {
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
    timerSeconds = 0;
    updateTimerDisplay();
    timerInterval = setInterval(() => { timerSeconds++; updateTimerDisplay(); }, 1000);
}

function stopTimer() {
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
}

function updateTimerDisplay() {
    const m = Math.floor(timerSeconds / 60).toString().padStart(2, "0");
    const s = (timerSeconds % 60).toString().padStart(2, "0");
    document.getElementById("timer").textContent = `${m}:${s}`;
}

// ── Helpers ─────────────────────────────────────────────────────────

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
}

function addRecordingToList(name, size) {
    const list = document.getElementById("recList");
    const empty = list.querySelector(".empty");
    if (empty) empty.remove();
    const item = document.createElement("div");
    item.className = "rec-item";
    item.innerHTML = `<span class="name">${name}</span><span class="size">${formatBytes(size)}</span>`;
    list.appendChild(item);
}

// ── Device Test ─────────────────────────────────────────────────────

async function testDevice(selectId, meterId, levelId, btnEl) {
    const deviceId = document.getElementById(selectId).value;
    if (!deviceId) { log("Select a device first", "warn"); return; }

    log(`Testing ${selectId.replace("deviceSelect", "Device ")} for 5 seconds...`);
    btnEl.disabled = true;

    try {
        const stream = await getStream(deviceId, true);
        const a = setupAnalyser(stream, meterId, levelId);
        activeAnalysers = [a];
        updateMeters();

        setTimeout(() => {
            cancelAnimationFrame(meterRAF);
            stream.getTracks().forEach(t => t.stop());
            a.ctx.close();
            btnEl.disabled = false;
            activeAnalysers = [];

            const level = document.getElementById(levelId).textContent;
            const db = parseFloat(level);
            if (isNaN(db) || db < -60) {
                log("No signal detected! Check mic selection.", "err");
            } else if (db < -40) {
                log(`Low signal (${db} dB). Mic may be wrong or too far.`, "warn");
            } else {
                log(`Signal OK (${db} dB).`, "ok");
            }
        }, 5000);
    } catch (err) {
        log(`Test failed: ${err.message}`, "err");
        btnEl.disabled = false;
    }
}

// ── Session ─────────────────────────────────────────────────────────

async function newSession() {
    const label = document.getElementById("sessionLabel").value.trim();
    try {
        const resp = await fetch("/api/session/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ label }),
        });
        const result = await resp.json();
        document.getElementById("sessionName").textContent = result.session;
        document.getElementById("recList").innerHTML =
            '<div class="empty" style="color:var(--text-dim);font-size:13px;">No recordings yet.</div>';
        log(`Session created: ${result.session}`, "ok");
    } catch (err) {
        log(`Failed to create session: ${err.message}`, "err");
    }
}

// ── Phase Selection ─────────────────────────────────────────────────

document.querySelectorAll(".phase-controls button").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".phase-controls button").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentPhase = btn.dataset.phase;
        log(`Phase: ${currentPhase}`);
    });
});

// ── Event Bindings ──────────────────────────────────────────────────

document.getElementById("btnRefreshDevices").addEventListener("click", enumerateDevices);
document.getElementById("btnTestA").addEventListener("click", function() {
    testDevice("deviceSelectA", "meterA_raw", "levelA_raw", this);
});
document.getElementById("btnTestB").addEventListener("click", function() {
    testDevice("deviceSelectB", "meterB_raw", "levelB_raw", this);
});
document.getElementById("btnNewSession").addEventListener("click", newSession);
document.getElementById("btnRecord").addEventListener("click", broadcastStart);

// ── SSE — listen for broadcast start/stop commands ──────────────────

function connectSSE() {
    const es = new EventSource("/api/events");

    es.addEventListener("start", (e) => {
        const data = JSON.parse(e.data);
        log("Broadcast START received", "ok");
        // Apply any broadcaster-supplied settings before starting
        if (data.phase) {
            currentPhase = data.phase;
            document.querySelectorAll(".phase-controls button").forEach(b => {
                b.classList.toggle("active", b.dataset.phase === data.phase);
            });
        }
        if (data.speaker) document.getElementById("speakerName").value = data.speaker;
        if (!isRecording) startRecording();
    });

    es.addEventListener("stop", () => {
        log("Broadcast STOP received", "ok");
        if (window._localStop) window._localStop();
    });

    es.onerror = () => {
        log("SSE disconnected — reconnecting in 3s...", "warn");
        es.close();
        setTimeout(connectSSE, 3000);
    };
}

// ── Broadcast helpers ────────────────────────────────────────────────

async function broadcastStart() {
    const phase = document.getElementById("customPhase").value.trim() || currentPhase;
    const speaker = document.getElementById("speakerName").value.trim();
    await fetch("/api/command/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phase, speaker }),
    });
    // startRecording is triggered by the SSE "start" event that bounces back to all clients including us
}

async function broadcastStop() {
    await fetch("/api/command/stop", { method: "POST" });
    if (window._localStop) await window._localStop();
}

// ── Init ────────────────────────────────────────────────────────────

(async function init() {
    log("Audio Capture (Dual Device) initializing...");

    // Restore client name from localStorage
    const savedName = localStorage.getItem("clientName");
    if (savedName) document.getElementById("clientName").value = savedName;
    document.getElementById("clientName").addEventListener("change", () => {
        localStorage.setItem("clientName", document.getElementById("clientName").value.trim());
    });
    try {
        await fetch("/api/session/current");
        document.getElementById("serverStatus").textContent = "Server connected";
        document.getElementById("serverStatus").style.color = "#4ecca3";
        log("Server OK", "ok");
    } catch {
        document.getElementById("serverStatus").textContent = "Server offline";
        document.getElementById("serverStatus").style.color = "#e74c3c";
        log("Cannot reach server!", "err");
    }
    connectSSE();
    await enumerateDevices();
    await newSession();
})();
