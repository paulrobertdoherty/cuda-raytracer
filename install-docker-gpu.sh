#!/usr/bin/env bash
# install-docker-gpu.sh — install Docker + NVIDIA Container Toolkit on Debian.
# Idempotent: safe to re-run. Must be run with sudo (or as root).
#
# Usage:
#   sudo ./install-docker-gpu.sh
#
# After this finishes, log out and back in (or `newgrp docker`) so the
# non-root user's docker group membership takes effect, then run:
#   ./run-claude.sh

set -euo pipefail

log() { printf '[install] %s\n' "$*"; }
die() { printf '[install] error: %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "must be run as root (use sudo)"

# The non-root user to add to the docker group. When invoked via sudo,
# SUDO_USER holds the original user.
TARGET_USER="${SUDO_USER:-}"
[[ -n "$TARGET_USER" && "$TARGET_USER" != "root" ]] \
  || log "warning: no SUDO_USER; skipping docker group setup"

export DEBIAN_FRONTEND=noninteractive

# --- prerequisites -------------------------------------------------------
log "installing prerequisites"
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl gpg

install -m 0755 -d /etc/apt/keyrings

# --- Docker repo ---------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  log "adding Docker apt repository"
  curl -fsSL https://download.docker.com/linux/debian/gpg \
    -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc

  . /etc/os-release
  tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/debian
Suites: ${VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

  apt-get update
  log "installing Docker engine + plugins"
  apt-get install -y --no-install-recommends \
    docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
else
  log "docker already installed: $(docker --version)"
fi

# --- NVIDIA Container Toolkit repo --------------------------------------
NVIDIA_LIST=/etc/apt/sources.list.d/nvidia-container-toolkit.list
if [[ ! -f "$NVIDIA_LIST" ]]; then
  log "adding NVIDIA Container Toolkit apt repository"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    > "$NVIDIA_LIST"

  apt-get update
fi

if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  log "installing nvidia-container-toolkit"
  apt-get install -y --no-install-recommends nvidia-container-toolkit
else
  log "nvidia-container-toolkit already installed"
fi

# --- runtime + service ---------------------------------------------------
log "configuring Docker runtime for NVIDIA"
nvidia-ctk runtime configure --runtime=docker

log "restarting docker"
systemctl restart docker

# --- docker group --------------------------------------------------------
if [[ -n "$TARGET_USER" && "$TARGET_USER" != "root" ]]; then
  if ! id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx docker; then
    log "adding $TARGET_USER to docker group"
    usermod -aG docker "$TARGET_USER"
    log "NOTE: log out + back in (or run 'newgrp docker') for this to take effect"
  else
    log "$TARGET_USER already in docker group"
  fi
fi

# --- smoke test ----------------------------------------------------------
SMOKE_IMAGE=nvidia/cuda:12.2.0-base-ubuntu22.04
log "smoke test: pulling $SMOKE_IMAGE (first run downloads ~250 MB)"
docker pull "$SMOKE_IMAGE"

log "smoke test: docker run --gpus all $SMOKE_IMAGE nvidia-smi"
if docker run --rm --gpus all "$SMOKE_IMAGE" nvidia-smi; then
  log "success: GPU visible inside containers"
else
  die "GPU smoke test failed. Check 'nvidia-smi' on the host and review docker logs."
fi

log "done. next: run ./run-claude.sh (log out/in first if you were just added to the docker group)"
