#!/usr/bin/env python3
"""
Audio Capture Server
====================
Local Python server that serves a Chrome-based audio capture page.
Records parallel streams with WebRTC processing ON and OFF for A/B comparison.
Stores everything as raw WAV in a timestamped session folder.

Usage:
    python server.py                    # start on http://localhost:8443
    python server.py --port 9000        # custom port
    python server.py --recordings-dir D:/my-recordings

Then open http://localhost:8443 in Chrome.
Chrome allows getUserMedia on localhost over plain HTTP — no cert needed.

Requirements:
    pip install flask flask-cors
"""

import argparse
import json
import os
import queue as _queue
import socket
import ssl
import struct
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path
from threading import Lock

# ── Check / install dependencies ─────────────────────────────────────

def ensure_deps():
    try:
        import flask
        import flask_cors
        import cryptography
    except ImportError:
        print("[setup] Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "flask", "flask-cors", "cryptography", "-q"
        ])

ensure_deps()

from flask import Flask, request, send_from_directory, jsonify, Response
from flask_cors import CORS


# ── Configuration ────────────────────────────────────────────────────

DEFAULT_PORT = 8080
CERT_FILE = "cert.pem"
KEY_FILE = "key.pem"

app = Flask(__name__)
CORS(app)

# Global state
recordings_dir = None
current_session = None
session_lock = Lock()

# SSE broadcast state
_sse_clients = []
_sse_lock = Lock()


def broadcast(event, data):
    """Push an SSE event to all connected clients."""
    msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(msg)
            except Exception:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


# ── SSL certificate generation ───────────────────────────────────────

def get_local_ip():
    """Return the LAN IP of this machine (hotspot/WiFi interface)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return None


def generate_self_signed_cert(cert_path, key_path, extra_ip=None):
    """Generate a self-signed SSL cert using the cryptography library."""
    if os.path.exists(cert_path) and os.path.exists(key_path) and not extra_ip:
        return True

    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509 import IPAddress, DNSName
    import ipaddress, datetime as dt

    ips = [ipaddress.IPv4Address("127.0.0.1")]
    if extra_ip:
        try:
            ips.append(ipaddress.IPv4Address(extra_ip))
        except ValueError:
            pass

    san_label = ", ".join(str(ip) for ip in ips)
    print(f"[ssl] Generating self-signed certificate (IPs: {san_label})...")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    san = x509.SubjectAlternativeName(
        [DNSName("localhost")] + [IPAddress(ip) for ip in ips]
    )
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + dt.timedelta(days=365))
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )

    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print("[ssl] Certificate generated.")
    return True


# ── Session management ───────────────────────────────────────────────

def create_session(label=""):
    """Create a new recording session folder."""
    global current_session
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"session_{ts}"
    if label:
        safe_label = "".join(c if c.isalnum() or c in "-_ " else "" for c in label).strip()
        name += f"_{safe_label}"
    session_path = os.path.join(recordings_dir, name)
    os.makedirs(session_path, exist_ok=True)

    current_session = {
        "name": name,
        "path": session_path,
        "started": datetime.now().isoformat(),
        "recordings": [],
    }
    # Write session metadata
    with open(os.path.join(session_path, "session.json"), "w") as f:
        json.dump({
            "name": name,
            "started": current_session["started"],
            "label": label,
        }, f, indent=2)

    print(f"[session] Created: {name}")
    return current_session


# ── WAV conversion ───────────────────────────────────────────────────

def webm_to_wav(webm_path, wav_path):
    """Convert webm/opus to raw WAV using ffmpeg."""
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", webm_path,
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "1",
            wav_path
        ], check=True, capture_output=True, timeout=30)
        return True
    except Exception as e:
        print(f"[convert] Failed: {e}")
        return False


def float32_pcm_to_wav(pcm_data, wav_path, sample_rate=48000, channels=1):
    """Convert raw Float32 PCM bytes to WAV."""
    n_samples = len(pcm_data) // 4  # 4 bytes per float32
    floats = struct.unpack(f"<{n_samples}f", pcm_data)

    # Convert float32 [-1, 1] to int16
    int16_samples = []
    for f in floats:
        clamped = max(-1.0, min(1.0, f))
        int16_samples.append(int(clamped * 32767))

    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(int16_samples)}h", *int16_samples))


# ── Routes: Static ───────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "capture.html")


@app.route("/capture.js")
def serve_js():
    return send_from_directory(".", "capture.js")


# ── Routes: API ──────────────────────────────────────────────────────

@app.route("/api/devices", methods=["GET"])
def list_devices():
    """Client-side enumeration happens in JS; this is a placeholder."""
    return jsonify({"status": "enumerate in browser"})


@app.route("/api/session/start", methods=["POST"])
def start_session():
    data = request.get_json() or {}
    label = data.get("label", "")
    with session_lock:
        session = create_session(label)
    return jsonify({"session": session["name"], "path": session["path"]})


@app.route("/api/session/current", methods=["GET"])
def get_current_session():
    with session_lock:
        if current_session:
            return jsonify({
                "session": current_session["name"],
                "recordings": len(current_session["recordings"]),
            })
        return jsonify({"session": None})


@app.route("/api/upload", methods=["POST"])
def upload_recording():
    """
    Receive a recorded audio blob from the browser.
    Expects multipart form with:
      - file: the audio blob
      - metadata: JSON with source, processing, phase, etc.
    """
    with session_lock:
        if not current_session:
            create_session("auto")

    f = request.files.get("file")
    meta_str = request.form.get("metadata", "{}")

    try:
        meta = json.loads(meta_str)
    except json.JSONDecodeError:
        meta = {}

    if not f:
        return jsonify({"error": "no file"}), 400

    # Build filename
    phase = meta.get("phase", "unknown")
    source = meta.get("source", "unknown")
    processing = meta.get("processing", "unknown")
    speaker = meta.get("speaker", "")
    client_name = meta.get("clientName", "")
    device_label = meta.get("deviceLabel", "")

    # Prefer the client-reported recording start time so the filename reflects
    # when audio capture actually began, not when the upload arrived here.
    recording_started_at = meta.get("recordingStartedAt", "")
    try:
        client_dt = datetime.fromisoformat(recording_started_at.replace("Z", "+00:00"))
        ts = client_dt.strftime("%H%M%S")
    except (ValueError, AttributeError):
        ts = datetime.now().strftime("%H%M%S")

    parts = [phase, source, processing]
    if speaker:
        parts.insert(1, speaker)
    if client_name:
        parts.insert(0, client_name)
    base_name = "_".join(parts)
    base_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in base_name)

    session_path = current_session["path"]

    # Save the original blob (webm/opus)
    orig_ext = "webm"
    content_type = f.content_type or ""
    if "wav" in content_type:
        orig_ext = "wav"
    elif "ogg" in content_type:
        orig_ext = "ogg"

    orig_name = f"{base_name}_{ts}.{orig_ext}"
    orig_path = os.path.join(session_path, orig_name)
    f.save(orig_path)
    file_size = os.path.getsize(orig_path)

    print(f"[upload] Saved: {orig_name} ({file_size:,} bytes)")

    # Convert to WAV
    wav_name = f"{base_name}_{ts}.wav"
    wav_path = os.path.join(session_path, wav_name)

    if orig_ext != "wav":
        if webm_to_wav(orig_path, wav_path):
            print(f"[convert] → {wav_name}")
        else:
            wav_name = None
            wav_path = None

    # Save metadata alongside
    meta_out = {
        **meta,
        "original_file": orig_name,
        "wav_file": wav_name,
        "timestamp": datetime.now().isoformat(),
        "recording_started_at": recording_started_at,   # client wall-clock, ms precision
        "file_size_bytes": file_size,
        "device_label": device_label,
    }
    meta_path = os.path.join(session_path, f"{base_name}_{ts}_meta.json")
    with open(meta_path, "w") as mf:
        json.dump(meta_out, mf, indent=2)

    with session_lock:
        current_session["recordings"].append(meta_out)

    return jsonify({
        "status": "ok",
        "original": orig_name,
        "wav": wav_name,
        "size": file_size,
    })


@app.route("/api/upload-pcm", methods=["POST"])
def upload_pcm():
    """
    Receive raw PCM float32 data from AudioWorklet.
    This bypasses the MediaRecorder/Opus codec entirely.
    """
    with session_lock:
        if not current_session:
            create_session("auto")

    meta_str = request.args.get("metadata", "{}")
    try:
        meta = json.loads(meta_str)
    except json.JSONDecodeError:
        meta = {}

    pcm_data = request.get_data()
    if not pcm_data:
        return jsonify({"error": "no data"}), 400

    phase = meta.get("phase", "unknown")
    source = meta.get("source", "unknown")
    speaker = meta.get("speaker", "")
    client_name = meta.get("clientName", "")
    sample_rate = meta.get("sampleRate", 48000)

    recording_started_at = meta.get("recordingStartedAt", "")
    try:
        client_dt = datetime.fromisoformat(recording_started_at.replace("Z", "+00:00"))
        ts = client_dt.strftime("%H%M%S")
    except (ValueError, AttributeError):
        ts = datetime.now().strftime("%H%M%S")

    parts = [phase, source, "pcm_raw"]
    if speaker:
        parts.insert(1, speaker)
    if client_name:
        parts.insert(0, client_name)
    base_name = "_".join(parts)
    base_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in base_name)

    session_path = current_session["path"]
    wav_name = f"{base_name}_{ts}.wav"
    wav_path = os.path.join(session_path, wav_name)

    float32_pcm_to_wav(pcm_data, wav_path, sample_rate=sample_rate)
    file_size = os.path.getsize(wav_path)
    print(f"[pcm] Saved: {wav_name} ({file_size:,} bytes, {len(pcm_data)//4} samples)")

    meta_out = {
        **meta,
        "wav_file": wav_name,
        "timestamp": datetime.now().isoformat(),
        "recording_started_at": recording_started_at,   # client wall-clock, ms precision
        "file_size_bytes": file_size,
        "sample_rate": sample_rate,
        "format": "pcm_float32_to_wav",
    }
    meta_path = os.path.join(session_path, f"{base_name}_{ts}_meta.json")
    with open(meta_path, "w") as mf:
        json.dump(meta_out, mf, indent=2)

    with session_lock:
        current_session["recordings"].append(meta_out)

    return jsonify({"status": "ok", "wav": wav_name, "size": file_size})


@app.route("/api/events")
def sse_stream():
    """Server-Sent Events stream — clients subscribe here for broadcast commands."""
    q = _queue.Queue(maxsize=20)
    with _sse_lock:
        _sse_clients.append(q)

    def generate():
        try:
            yield "event: connected\ndata: {}\n\n"
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield msg
                except _queue.Empty:
                    yield ": keepalive\n\n"  # prevent proxy/browser timeout
        finally:
            with _sse_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/command/start", methods=["POST"])
def cmd_start():
    data = request.get_json() or {}
    broadcast("start", data)
    return jsonify({"status": "broadcasted", "clients": len(_sse_clients)})


@app.route("/api/command/stop", methods=["POST"])
def cmd_stop():
    broadcast("stop", {})
    return jsonify({"status": "broadcasted", "clients": len(_sse_clients)})


@app.route("/api/recordings", methods=["GET"])
def list_recordings():
    """List all sessions and their recordings."""
    sessions = []
    for d in sorted(Path(recordings_dir).iterdir()):
        if d.is_dir():
            meta_path = d / "session.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            else:
                meta = {"name": d.name}
            files = [f.name for f in d.iterdir() if f.suffix in (".wav", ".webm")]
            meta["files"] = files
            meta["count"] = len(files)
            sessions.append(meta)
    return jsonify(sessions)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    global recordings_dir

    parser = argparse.ArgumentParser(description="Audio Capture Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--recordings-dir", type=str, default=None,
                        help="Where to store recordings (default: ./recordings)")
    parser.add_argument("--ssl", action="store_true",
                        help="Enable HTTPS with self-signed cert (usually not needed)")
    args = parser.parse_args()

    # Set up recordings directory
    server_dir = os.path.dirname(os.path.abspath(__file__))
    recordings_dir = args.recordings_dir or os.path.join(server_dir, "recordings")
    os.makedirs(recordings_dir, exist_ok=True)
    print(f"[config] Recordings directory: {recordings_dir}")

    # Change to server directory so static files are found
    os.chdir(server_dir)

    # SSL setup — only if explicitly requested
    # Chrome allows getUserMedia on localhost over plain HTTP, so HTTPS is optional.
    lan_ip = get_local_ip()
    ssl_ctx = None
    protocol = "http"
    if args.ssl:
        cert_ok = generate_self_signed_cert(CERT_FILE, KEY_FILE, extra_ip=lan_ip)
        if cert_ok:
            try:
                ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ssl_ctx.load_cert_chain(CERT_FILE, KEY_FILE)
                protocol = "https"
            except Exception as e:
                print(f"[ssl] Failed to load cert: {e}")
                print("[ssl] Falling back to HTTP.")

    print(f"\n{'='*60}")
    print(f"  Audio Capture Server")
    print(f"  Local:   {protocol}://localhost:{args.port}")
    if lan_ip and protocol == "https":
        print(f"  LAN:     {protocol}://{lan_ip}:{args.port}  ← share this with others")
        print(f"  (Others: click 'Advanced → Proceed anyway' on the cert warning)")
    elif lan_ip:
        print(f"  LAN:     http://{lan_ip}:{args.port}  (mic blocked — run with --ssl for LAN use)")
    print(f"  Recordings: {recordings_dir}")
    print(f"{'='*60}\n")

    app.run(
        host="0.0.0.0",
        port=args.port,
        ssl_context=ssl_ctx,
        debug=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
