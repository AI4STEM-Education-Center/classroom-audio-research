Audio Capture Server — Quick Start
====================================

1. INSTALL (one time):
   pip install flask flask-cors

2. RUN:
   cd audio-capture-server
   python server.py

3. OPEN in Chrome:
   http://localhost:8443
   (Chrome allows mic access on localhost over plain HTTP — no cert needed.
    If you need HTTPS for some reason, run: python server.py --ssl)

4. USE:
   - Select your headset mic from the dropdown
   - Click "Test Selected Device" to verify signal
   - Select a phase (Setup, Enrollment, etc.)
   - Click START RECORDING — three parallel captures begin:
     A) RAW:       WebRTC processing OFF → webm/opus → auto-converted to WAV
     B) PROCESSED: WebRTC processing ON  → webm/opus → auto-converted to WAV
     C) PCM RAW:   Float32 PCM via ScriptProcessor → direct to WAV (no codec)
   - Click STOP — all three files are uploaded and saved

5. RECORDINGS:
   All files saved to: audio-capture-server/recordings/<session>/
   Each recording creates:
     - .webm file (original browser output)
     - .wav file (converted, 48kHz 16-bit mono)
     - _meta.json (metadata: phase, source, processing, device, timestamp)

6. OPTIONS:
   python server.py --port 9000                 # different port
   python server.py --recordings-dir D:/data    # custom output folder
   python server.py --no-ssl                    # skip HTTPS (mic may not work)

7. REQUIREMENTS:
   - Python 3.8+
   - Chrome (or Chromium-based browser)
   - ffmpeg (for webm→wav conversion)
   - openssl (for self-signed cert, usually pre-installed)

NOTES:
- HTTPS is required for getUserMedia() to work in Chrome.
  The server auto-generates a self-signed cert on first run.
- The PCM channel bypasses the Opus codec entirely, giving you
  the true raw audio as the browser's audio pipeline sees it.
- Comparing RAW vs PROCESSED from the same session shows you
  exactly what Chrome's WebRTC processing does to the signal.
