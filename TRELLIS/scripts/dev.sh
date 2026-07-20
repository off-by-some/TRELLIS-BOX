#!/bin/bash

# Quick development mode launcher
# Directly uses docker compose without all the checks

set -e
export DOCKER_BUILDKIT=1

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi
if [ -f ".trellis.env" ]; then
    set -a
    source .trellis.env
    set +a
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

is_portable_runtime() {
    case "${TRELLIS_DEVICE:-auto}" in
        cpu|mps)
            return 0
            ;;
    esac

    if [ "$(uname -s)" = "Darwin" ]; then
        return 0
    fi

    return 1
}

detect_cpu_threads() {
    if [ -n "${TRELLIS_CPU_THREADS:-}" ]; then
        echo "$TRELLIS_CPU_THREADS"
        return
    fi

    if [ "$(uname -s)" = "Darwin" ] && command -v sysctl > /dev/null 2>&1; then
        sysctl -n hw.physicalcpu 2>/dev/null && return
    fi

    getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4
}

configure_cpu_threads() {
    local cpu_threads
    cpu_threads="$(detect_cpu_threads)"
    export TRELLIS_CPU_THREADS=${TRELLIS_CPU_THREADS:-$cpu_threads}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$cpu_threads}
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$cpu_threads}
    export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-$cpu_threads}
    export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-$cpu_threads}
    export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-$cpu_threads}
}

start_portable_local() {
    APP_PORT=${APP_PORT:-8501}
    HOST_PORT=${HOST_PORT:-8501}
    configure_cpu_threads
    export TRELLIS_DEVICE=${TRELLIS_DEVICE:-cpu}
    export SPARSE_BACKEND=${SPARSE_BACKEND:-torch}
    export ATTN_BACKEND=${ATTN_BACKEND:-sdpa}
    if [ -z "${TRELLIS_OUTPUT_DIR}" ] || [[ "${TRELLIS_OUTPUT_DIR}" == /app/* ]]; then
        export TRELLIS_OUTPUT_DIR=./outputs
    fi
    if [ -z "${U2NET_HOME}" ] || [[ "${U2NET_HOME}" == /app/* ]]; then
        export U2NET_HOME=./rembg_cache
    fi
    export STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}
    export STREAMLIT_SERVER_HEADLESS=${STREAMLIT_SERVER_HEADLESS:-true}

    mkdir -p "$TRELLIS_OUTPUT_DIR"
    mkdir -p "$U2NET_HOME"

    PYTHON_BIN=${PYTHON_BIN:-python3}
    if ! command -v "$PYTHON_BIN" > /dev/null 2>&1; then
        print_error "Python executable not found: $PYTHON_BIN"
        print_error "Create a macOS environment first, then try again:"
        print_error "  python3.10 -m venv .venv && source .venv/bin/activate"
        print_error "  pip install -r requirements.macos.txt"
        exit 1
    fi

    if ! "$PYTHON_BIN" -c "import streamlit" > /dev/null 2>&1; then
        print_error "Streamlit is not installed in this Python environment."
        print_error "Install portable dependencies first:"
        print_error "  pip install -r requirements.macos.txt"
        exit 1
    fi

    print_status "Starting portable local development mode"
    print_warning "Portable mode is experimental, mesh-only, and can be very slow."
    print_status "Runtime: TRELLIS_DEVICE=${TRELLIS_DEVICE}, SPARSE_BACKEND=${SPARSE_BACKEND}, ATTN_BACKEND=${ATTN_BACKEND}"
    print_status "CPU threads: ${TORCH_NUM_THREADS}"
    print_status "Access at: http://localhost:${HOST_PORT}"
    print_status "Press Ctrl+C to stop"
    echo ""

    exec "$PYTHON_BIN" -m streamlit run app.py \
        --server.port "${HOST_PORT}" \
        --server.address "${STREAMLIT_SERVER_ADDRESS}" \
        --server.headless "${STREAMLIT_SERVER_HEADLESS}"
}

echo "🔥 TRELLIS Development Mode"
echo "============================"
echo ""

if is_portable_runtime; then
    start_portable_local
fi

# Detect docker compose command
COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    print_status "Using docker-compose (standalone)"
elif docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
    print_status "Using docker compose (plugin)"
else
    print_error "Neither 'docker-compose' nor 'docker compose' found"
    print_error "Install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    print_status "No .env file found (using defaults)"
fi

print_status "Starting development mode with hot reloading..."
print_status "Files mounted:"
print_status "  - app.py (hot-reloadable)"
print_status "  - webui/ (hot-reloadable)"
print_status "  - docs/ (hot-reloadable)"
print_status "  - assets/ (hot-reloadable)"
echo ""
print_status "Press Ctrl+C to stop"
echo ""

# Stop existing containers
$COMPOSE_CMD down > /dev/null 2>&1

# Start in development mode
$COMPOSE_CMD -f docker-compose.yml -f docker-compose.dev.yml up --build

print_success "Development container stopped"
