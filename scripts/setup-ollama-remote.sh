#!/bin/bash
# Run on a second machine to host the LLM pass there: installs Ollama, binds it to one address
# (your Tailscale IP, so it is reachable only inside your tailnet), keeps the model resident, pulls it.
#   bash setup-ollama-remote.sh <bind-ip> [model]
# Then on the transcription machine set OLLAMA_URL=http://<bind-ip>:11434 and LLM_MODEL=<model> in ~/.config/audio-notes/env.
set -e
BIND=${1:?usage: setup-ollama-remote.sh <bind-ip> [model]}
MODEL=${2:-qwen3.6:35b-a3b}
curl -fsSL https://ollama.com/install.sh | sh
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment=OLLAMA_HOST=%s\nEnvironment=OLLAMA_KEEP_ALIVE=-1\nEnvironment=OLLAMA_FLASH_ATTENTION=1\nEnvironment=OLLAMA_KV_CACHE_TYPE=q8_0\n' "$BIND" | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 3
OLLAMA_HOST=$BIND ollama pull "$MODEL"
