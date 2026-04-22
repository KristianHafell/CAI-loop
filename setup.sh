#!/usr/bin/env zsh
# =============================================================
# setup.sh  —  Prepare environment for the CAI critique-revision loop
# Usage: source setup.sh   (or  . setup.sh)
# =============================================================

set -e  # exit on first error

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   CAI Loop — Environment Setup           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Load .env ─────────────────────────────────────────────
echo "▶ Loading .env …"
if [[ ! -f .env ]]; then
  echo "  ✗ .env not found. Make sure it is in the same folder as this script."
  return 1 2>/dev/null || exit 1
fi

if grep -qE '^[[:space:]]*HF_TOKEN=' .env; then
  # Load environment variables from .env safely
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a

  # Extract HF_TOKEN value (first match), trim whitespace and surrounding quotes
  HF_TOKEN_VAL=$(grep -E '^[[:space:]]*HF_TOKEN=' .env | head -n1 | cut -d '=' -f2- | sed -E 's/^[[:space:]]*//;s/[[:space:]]*$//' | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")

  if [ -n "$HF_TOKEN_VAL" ]; then
    # Mask value for safe display (show first 4 and last 4 chars when long)
    if [ ${#HF_TOKEN_VAL} -gt 8 ]; then
      first=${HF_TOKEN_VAL:0:4}
      last=${HF_TOKEN_VAL: -4}
      MASKED="$first...$last"
    else
      MASKED="$HF_TOKEN_VAL"
    fi
    echo "  ✓ HF_TOKEN loaded (value: $MASKED)"
  else
    echo "  ✓ HF_TOKEN found but empty"
  fi
else
  echo "  ✗ HF_TOKEN not found in .env"
fi

# ── 2. Python version check ───────────────────────────────────
echo ""
echo "▶ Checking Python version …"
PY_VERSION=$(python --version 2>&1)
PY_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
PY_MAJOR=$(python -c "import sys; print(sys.version_info.major)")

if [[ "$PY_MAJOR" -lt 3 || "$PY_MINOR" -lt 8 ]]; then
  echo "  ✗ Requires Python 3.8+, found: $PY_VERSION"
  echo "    Make sure 'python' points to your 3.12 install."
  return 1 2>/dev/null || exit 1
fi
echo "  ✓ $PY_VERSION"

# ── 3. Python deps ────────────────────────────────────────────
echo ""
echo "▶ Installing Python dependencies …"
python -m pip install --quiet datasets requests
echo "  ✓ datasets, requests installed"

# ── 3. Check Ollama is running ────────────────────────────────
echo ""
echo "▶ Checking Ollama …"
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
  echo "  ✗ Ollama is not running. Start it with: ollama serve"
  return 1 2>/dev/null || exit 1
fi
echo "  ✓ Ollama is running"

# ── 4. Pull models if not already present ────────────────────
echo ""
echo "▶ Checking models …"

PULLED=$(curl -s http://localhost:11434/api/tags | python -c \
  "import sys,json; models=json.load(sys.stdin).get('models',[]); print(' '.join(m['name'] for m in models))")

if echo "$PULLED" | grep -q "llama3.1"; then
  echo "  ✓ llama3.1:8b already pulled"
else
  echo "  ↓ Pulling llama3.1:8b (this may take a while) …"
  ollama pull llama3.1:8b
  echo "  ✓ llama3.1:8b ready"
fi

if echo "$PULLED" | grep -q "llama-guard3"; then
  echo "  ✓ llama-guard3:8b already pulled"
else
  echo "  ↓ Pulling llama-guard3:8b …"
  ollama pull llama-guard3:8b
  echo "  ✓ llama-guard3:8b ready"
fi

# ── 5. HuggingFace login ──────────────────────────────────────
echo ""
echo "▶ Logging in to HuggingFace …"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null \
  || pip install --quiet huggingface_hub && huggingface-cli login --token "$HF_TOKEN"
echo "  ✓ HuggingFace authenticated"

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   All set! Run the experiment with:      ║"
echo "║                                          ║"
echo "║   python cai_loop.py --n 3 --loops 1     ║"
echo "╚══════════════════════════════════════════╝"
echo ""