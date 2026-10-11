#!/bin/bash
# One-time setup on the transcription machine (Ubuntu, NVIDIA GPU, ffmpeg, Python 3.12).
#   bash scripts/setup.sh
set -e
cd "$(dirname "$0")/.."
command -v ffmpeg >/dev/null || { echo "install ffmpeg first (apt install ffmpeg)"; exit 1; }
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# DeepFilterNet3 static binary for noise removal (no Python deps)
if [ ! -x bin/deep-filter ]; then
  mkdir -p bin
  curl -L -o bin/deep-filter https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-x86_64-unknown-linux-musl
  chmod +x bin/deep-filter
fi
# config: the diarization model is gated, so a Hugging Face token is needed once
mkdir -p ~/.config/audio-notes
if [ ! -f ~/.config/audio-notes/env ]; then
  cat > ~/.config/audio-notes/env <<'ENV'
# After editing: systemctl --user restart audio-notes
# 1. Accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1
# 2. Create a read token at https://huggingface.co/settings/tokens and paste it here:
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# systemd reads this file literally: one KEY=value per line, no trailing comments.
# window (default) or turns
ASR_MODE=window
# qwen (default) or whisper
# ASR_BACKEND=qwen
# Ollama model for the Hinglish + English + summary pass
# LLM_MODEL=qwen3.5:9b
# point at another machine to keep this GPU for speech
# OLLAMA_URL=http://127.0.0.1:11434
# NOTES_DIR=/home/you/notes/audio
# More than one person: your own tailnet login (tailscale status --json | jq -r '.User[(.Self.UserID|tostring)].LoginName').
# Anyone else on the tailnet gets their own notes, voices and settings under USERS_DIR.
# OWNER_LOGIN=you@github
# USERS_DIR=/home/you/notes/audio-users
ENV
  chmod 600 ~/.config/audio-notes/env
  echo ">> edit ~/.config/audio-notes/env and paste your Hugging Face token"
fi
# user service: starts on boot (needs: loginctl enable-linger $USER)
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/audio-notes.service <<UNIT
[Unit]
Description=kaunbola transcription server
After=network.target

[Service]
WorkingDirectory=$PWD
EnvironmentFile=-%h/.config/audio-notes/env
ExecStart=$PWD/.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8470
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now audio-notes
echo ">> running on http://127.0.0.1:8470"
echo ">> to reach it from your phone over Tailscale (HTTPS is required for the microphone):"
echo "   tailscale serve --bg --https=8446 http://127.0.0.1:8470"
