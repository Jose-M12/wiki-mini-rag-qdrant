#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
cd "$PROJECT_DIR"

# ── Venv check ──────────────────────────────────────────────────
if [ ! -f "ragsetup/bin/python" ]; then
    echo "ERROR: Virtual environment 'ragsetup' not found. Run setup first." >&2
    exit 1
fi

# ── Qdrant config ──────────────────────────────────────────────
QDRANT_CONTAINER_NAME="wiki-mini-qdrant"
QDRANT_IMAGE="qdrant/qdrant:latest"
QDRANT_PORT="6333"
QDRANT_DATA_DIR="./qdrant_data"

start_qdrant() {
    if docker ps --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER_NAME}$"; then
        echo "Qdrant container already running."
        return
    fi
    if docker ps -a --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER_NAME}$"; then
        docker rm "$QDRANT_CONTAINER_NAME" >/dev/null
    fi
    mkdir -p -m 0700 "$QDRANT_DATA_DIR"
    echo "Starting Qdrant container…"
    docker run -d \
        --name "$QDRANT_CONTAINER_NAME" \
        -p 127.0.0.1:${QDRANT_PORT}:6333 \
        -v "$(pwd)/${QDRANT_DATA_DIR}:/qdrant/storage" \
        "$QDRANT_IMAGE"
}

stop_qdrant() {
    if docker ps --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER_NAME}$"; then
        echo "Stopping Qdrant container…"
        docker stop "$QDRANT_CONTAINER_NAME" >/dev/null
    fi
}

wait_for_qdrant() {
    local max_attempts=30
    local attempt=1
    echo "Waiting for Qdrant…"
    until curl -s "http://127.0.0.1:${QDRANT_PORT}/health" | grep -q 'ok'; do
        sleep 1
        attempt=$((attempt + 1))
        if [ "$attempt" -gt "$max_attempts" ]; then
            echo "ERROR: Qdrant not ready in time." >&2
            stop_qdrant
            exit 1
        fi
    done
    echo "Qdrant ready."
}

KEEP_RUNNING=false
PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-running)
            KEEP_RUNNING=true
            shift
            ;;
        *)
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ ${#PYTHON_ARGS[@]} -eq 0 ]; then
    echo "Usage: $0 [--keep-running] <wiki_mini_cli.py args>" >&2
    exit 1
fi

start_qdrant
wait_for_qdrant

set +e
ragsetup/bin/python wiki_mini_cli.py "${PYTHON_ARGS[@]}"
EXIT_CODE=$?
set -e

if [ "$KEEP_RUNNING" = false ]; then
    stop_qdrant
fi

exit $EXIT_CODE
