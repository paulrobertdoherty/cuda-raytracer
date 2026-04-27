#!/usr/bin/env bash
# run-claude.sh — launch Claude Code inside a Docker container with
# --dangerously-skip-permissions, isolated from the host.
#
# Modeled on https://github.com/icanhasjonas/run-claude-docker, tailored for
# this CUDA ray tracer: CUDA 12.2 devel base, headless build only, no MCP
# servers, --gpus all. The host runs the GUI binary; the container is for
# Claude + nvcc/cmake/gdb/valgrind.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_NAME="$(basename "$REPO_DIR")"
IMAGE="${REPO_NAME}-claude:latest"
CONTAINER_DEFAULT="${REPO_NAME}-claude"
CUDA_BASE_TAG="12.2.0-devel-ubuntu22.04"
CUDA_PROBE_TAG="12.2.0-base-ubuntu22.04"

CONTAINER="$CONTAINER_DEFAULT"
DO_BUILD=0
DO_REBUILD=0
SAFE=0
ONE_SHOT=0
PASSTHRU=()

die() { printf 'run-claude: %s\n' "$*" >&2; exit 1; }
log() { printf 'run-claude: %s\n' "$*" >&2; }

print_help() {
  cat <<EOF
Usage: ./run-claude.sh [flags] [-- <args passed to claude>]

Runs Claude Code inside a Docker container with --dangerously-skip-permissions,
isolated from your host filesystem. Repo is mounted at /workspace. The
container gets --gpus all so CUDA builds work. Headless build only; run the
GUI binary on the host.

No flags:       drop into an interactive bash shell in the container
                (claude is on \$PATH; run it manually).
Positional:     ./run-claude.sh "fix the BVH kernel"
                Everything after '--' (or the first non-flag positional) is
                forwarded to claude.

Flags:
  --build       build image only, then exit
  --rebuild     remove container + image, rebuild, then run
  --safe        drop --dangerously-skip-permissions
  --rm          one-shot: docker run --rm (no container reuse)
  -n, --name N  override container name (default: ${CONTAINER_DEFAULT})
  -h, --help    this help

Image:      ${IMAGE}
Container:  ${CONTAINER_DEFAULT}
Workspace:  ${REPO_DIR} -> /workspace

Persisted on host (bind-mounted):
  ~/.claude                -> /home/dev/.claude     (rw, auth/sessions)
  ~/.claude.json           -> /home/dev/.claude.json (rw, if present)
  ~/.gitconfig             -> /home/dev/.gitconfig   (ro, if present)
SSH keys are intentionally NOT mounted.

The container runs as user 'dev' (UID/GID matched to your host user) because
claude --dangerously-skip-permissions refuses to run as root.

Note: docker exec on a running container uses the image the container started
with. After --rebuild or an image update, stop the container (or use --rebuild)
so the new container picks up the new image.
EOF
}

# --- arg parsing ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)   DO_BUILD=1; shift ;;
    --rebuild) DO_REBUILD=1; shift ;;
    --safe)    SAFE=1; shift ;;
    --rm)      ONE_SHOT=1; shift ;;
    -n|--name)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      CONTAINER="$2"; shift 2 ;;
    -h|--help) print_help; exit 0 ;;
    --)        shift; PASSTHRU+=("$@"); break ;;
    -*)        die "unknown flag: $1 (try --help)" ;;
    *)         PASSTHRU+=("$@"); break ;;
  esac
done

# --- prereqs -------------------------------------------------------------
check_prereqs() {
  command -v docker >/dev/null 2>&1 \
    || die "docker not found. Install: https://docs.docker.com/engine/install/"

  if ! docker run --rm --gpus all "nvidia/cuda:${CUDA_PROBE_TAG}" \
         nvidia-smi >/dev/null 2>&1; then
    die "GPU not accessible to Docker. Install nvidia-container-toolkit:
  sudo apt install nvidia-container-toolkit
  sudo systemctl restart docker"
  fi
}

ensure_host_dirs() {
  # Pre-create so Docker doesn't create it root-owned on first bind-mount.
  mkdir -p "$HOME/.claude"
}

# --- image build ---------------------------------------------------------
image_exists() { docker image inspect "$IMAGE" >/dev/null 2>&1; }

build_image() {
  local tmp rc=0
  tmp="$(mktemp -d)"

  cat >"$tmp/Dockerfile" <<DOCKERFILE
FROM nvidia/cuda:${CUDA_BASE_TAG}
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \\
      ca-certificates curl git gpg gpg-agent pkg-config build-essential \\
      ninja-build gdb valgrind less ripgrep jq python3 python3-pip sudo \\
 && rm -rf /var/lib/apt/lists/*

# Kitware APT for cmake >= 3.24 (project requires 3.24; jammy ships 3.22).
RUN curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \\
      | gpg --dearmor -o /usr/share/keyrings/kitware.gpg \\
 && echo "deb [signed-by=/usr/share/keyrings/kitware.gpg] https://apt.kitware.com/ubuntu/ jammy main" \\
      > /etc/apt/sources.list.d/kitware.list \\
 && apt-get update && apt-get install -y --no-install-recommends cmake \\
 && rm -rf /var/lib/apt/lists/*

# Node 20 + Claude Code.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \\
 && apt-get install -y --no-install-recommends nodejs \\
 && rm -rf /var/lib/apt/lists/* \\
 && npm install -g @anthropic-ai/claude-code

# beads (issue tracker used by this project's CLAUDE.md workflow).
ARG BEADS_VERSION=1.0.3
RUN curl -fsSL "https://github.com/gastownhall/beads/releases/download/v\${BEADS_VERSION}/beads_\${BEADS_VERSION}_linux_amd64.tar.gz" \\
      | tar -xz -C /tmp \\
 && install -m 0755 /tmp/bd /usr/local/bin/bd \\
 && rm -f /tmp/bd

# Non-root user matching host UID/GID. claude --dangerously-skip-permissions
# refuses to run as root. Passwordless sudo so apt-get etc. still works.
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd -g \${HOST_GID} dev \\
 && useradd -m -u \${HOST_UID} -g \${HOST_GID} -s /bin/bash dev \\
 && echo 'dev ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/dev \\
 && chmod 0440 /etc/sudoers.d/dev

USER dev
ENV HOME=/home/dev
WORKDIR /workspace
CMD ["bash"]
DOCKERFILE

  log "building image ${IMAGE}..."
  docker build \
    --build-arg "HOST_UID=$(id -u)" \
    --build-arg "HOST_GID=$(id -g)" \
    -t "$IMAGE" -f "$tmp/Dockerfile" "$tmp" || rc=$?
  rm -rf "$tmp"
  return $rc
}

ensure_image() {
  if (( DO_REBUILD )); then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    docker rmi "$IMAGE" >/dev/null 2>&1 || true
  fi
  if ! image_exists; then
    build_image
  fi
}

# --- container run -------------------------------------------------------
container_exists()  { docker container inspect "$CONTAINER" >/dev/null 2>&1; }
container_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" == "true" ]]
}

build_run_args() {
  # Populates global RUN_ARGS and CMD_ARGS.
  RUN_ARGS=( --gpus all -it
             --name "$CONTAINER"
             -v "$REPO_DIR:/workspace"
             -w /workspace
             -v "$HOME/.claude:/home/dev/.claude"
             -e "TERM=${TERM:-xterm-256color}" )

  (( ONE_SHOT )) && RUN_ARGS+=( --rm )

  [[ -f "$HOME/.claude.json" ]] \
    && RUN_ARGS+=( -v "$HOME/.claude.json:/home/dev/.claude.json" )
  [[ -f "$HOME/.gitconfig" ]] \
    && RUN_ARGS+=( -v "$HOME/.gitconfig:/home/dev/.gitconfig:ro" )

  local safe_flag="--dangerously-skip-permissions"
  (( SAFE )) && safe_flag=""

  if (( ${#PASSTHRU[@]} == 0 )); then
    CMD_ARGS=( bash )
  elif [[ -z "$safe_flag" ]]; then
    CMD_ARGS=( bash -lc 'claude "$@"' _ "${PASSTHRU[@]}" )
  else
    CMD_ARGS=( bash -lc 'claude --dangerously-skip-permissions "$@"' _ "${PASSTHRU[@]}" )
  fi
}

run_exec_in_running() {
  log "exec into running container ${CONTAINER}"
  if (( ${#PASSTHRU[@]} == 0 )); then
    exec docker exec -it "$CONTAINER" bash
  fi
  if (( SAFE )); then
    exec docker exec -it "$CONTAINER" \
      bash -lc 'claude "$@"' _ "${PASSTHRU[@]}"
  fi
  exec docker exec -it "$CONTAINER" \
    bash -lc 'claude --dangerously-skip-permissions "$@"' _ "${PASSTHRU[@]}"
}

warn_if_stale() {
  local img_id cont_img
  img_id="$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
  cont_img="$(docker inspect -f '{{.Image}}' "$CONTAINER" 2>/dev/null || true)"
  if [[ -n "$img_id" && -n "$cont_img" && "$img_id" != "$cont_img" ]]; then
    log "warning: container ${CONTAINER} was created from an older image."
    log "         run with --rebuild to pick up image changes."
  fi
}

# --- main ----------------------------------------------------------------
check_prereqs
ensure_host_dirs
ensure_image

if (( DO_BUILD )); then
  log "image built: ${IMAGE}"
  exit 0
fi

build_run_args

if (( ONE_SHOT )); then
  exec docker run "${RUN_ARGS[@]}" "$IMAGE" "${CMD_ARGS[@]}"
fi

if container_running; then
  warn_if_stale
  run_exec_in_running
fi

if container_exists; then
  warn_if_stale
  log "resuming container ${CONTAINER}"
  exec docker start -ai "$CONTAINER"
fi

log "creating container ${CONTAINER}"
exec docker run "${RUN_ARGS[@]}" "$IMAGE" "${CMD_ARGS[@]}"
