#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Benchmarking — one-shot setup script
#
# What it does:
#   1. Ensures Python 3.13 is available (installs via pyenv if needed)
#   2. Clones https://github.com/endee-io/hybrid_benchmarking  (skipped if
#      this script is already running from inside the repo)
#   3. Creates index-env  and  validation-env  with python3.13
#   4. pip-installs index-env.txt  and  validation-env.txt into each env
#
# Run from anywhere:
#   bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────

REPO_URL="https://github.com/endee-io/hybrid_benchmarking"
PYTHON_VERSION="3.13.0"

# ── colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
die()  { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

# ── Step 1: Python 3.13 ──────────────────────────────────────────────────────

log "Checking for Python 3.13..."

PYTHON_BIN=""

# Prefer a system/existing python3.13 binary
if command -v python3.13 &>/dev/null; then
    ver=$(python3.13 --version 2>&1)
    if [[ "$ver" == *"3.13"* ]]; then
        log "Found $ver — skipping pyenv install."
        PYTHON_BIN="python3.13"
    fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
    log "Python 3.13 not found — installing via pyenv."

    # ── system build deps ────────────────────────────────────────────────────
    log "Installing system build dependencies (requires sudo)..."
    sudo apt-get update -qq
    sudo apt-get install -y \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        curl \
        git

    # ── install pyenv ────────────────────────────────────────────────────────
    if [[ ! -d "$HOME/.pyenv" ]]; then
        log "Installing pyenv..."
        curl -fsSL https://pyenv.run | bash
    else
        log "pyenv directory already exists — skipping installer."
    fi

    # ── activate pyenv in this shell session ─────────────────────────────────
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

    # ── persist pyenv in ~/.bashrc if not already there ──────────────────────
    SHELL_RC="$HOME/.bashrc"
    if ! grep -q 'pyenv init' "$SHELL_RC" 2>/dev/null; then
        log "Adding pyenv initialisation to $SHELL_RC..."
        cat >> "$SHELL_RC" <<'BASHRC'

# pyenv — added by hybrid_benchmarking/setup.sh
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
BASHRC
        log "Run  source ~/.bashrc  (or open a new terminal) after this script."
    fi

    # ── install Python 3.13 via pyenv ────────────────────────────────────────
    # -s = skip if already installed; existing versions are left untouched
    log "Installing Python $PYTHON_VERSION via pyenv (this can take a few minutes)..."
    pyenv install -s "$PYTHON_VERSION"

    # Set 3.13 as the global default; other installed versions remain available
    pyenv global "$PYTHON_VERSION"

    PYTHON_BIN="python3.13"
    log "Python ready: $($PYTHON_BIN --version)"
fi

# ── Step 2: Clone repo ───────────────────────────────────────────────────────

# ── Step 2a: Ensure git is available ─────────────────────────────────────────

if ! command -v git &>/dev/null; then
    log "git not found — installing..."
    sudo apt-get update -qq
    sudo apt-get install -y git
fi

if ! "$PYTHON_BIN" -m ensurepip --version &>/dev/null 2>&1; then
    log "python3-venv not available — installing python3.13-venv..."
    sudo apt-get update -qq
    sudo apt-get install -y python3.13-venv
fi

# Detect if this script is already inside the repo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=""

if [[ -f "$SCRIPT_DIR/index-env.txt" && -f "$SCRIPT_DIR/validation-env.txt" ]]; then
    log "Running from inside the repo — skipping clone."
    REPO_DIR="$SCRIPT_DIR"
else
    TARGET_DIR="$(pwd)/hybrid_benchmarking"
    if [[ -d "$TARGET_DIR" ]]; then
        warn "Directory '$TARGET_DIR' already exists — skipping clone."
    else
        log "Cloning $REPO_URL (branch: qps_calculation_correction)..."
        git clone --branch qps_calculation_correction "$REPO_URL" "$TARGET_DIR"
    fi
    REPO_DIR="$TARGET_DIR"
fi

cd "$REPO_DIR"
log "Working in: $REPO_DIR"

# ── Step 3: Virtual environments ─────────────────────────────────────────────

for env in index-env validation-env; do
    if [[ -f "$env/bin/activate" ]]; then
        warn "Virtual environment '$env' already exists — skipping creation."
    else
        log "Creating $env with $($PYTHON_BIN --version)..."
        "$PYTHON_BIN" -m venv "$env"
    fi
done

# ── Step 4: Install requirements ─────────────────────────────────────────────

log "Installing index-env requirements from index-env.txt..."
# shellcheck source=/dev/null
source index-env/bin/activate
pip install --upgrade pip --quiet
pip install -r index-env.txt
deactivate
log "index-env ready."

log "Installing validation-env requirements from validation-env.txt..."
# shellcheck source=/dev/null
source validation-env/bin/activate
pip install --upgrade pip --quiet
pip install -r validation-env.txt
deactivate
log "validation-env ready."

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
log "Setup complete!"
echo ""
echo "  Repo:          $REPO_DIR"
echo "  Python:        $($PYTHON_BIN --version)"
echo ""
echo "  Activate envs:"
echo "    source $REPO_DIR/index-env/bin/activate      # indexing + querying (main.py)"
echo "    source $REPO_DIR/validation-env/bin/activate # metrics / validation"
echo ""
