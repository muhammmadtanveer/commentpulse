#!/bin/bash
# CommentPulse — one-command run script for Mac/Linux
# Usage:  bash run-mac.sh
cd "$(dirname "$0")" || exit 1

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed yet."
  echo "Install it from https://www.python.org/downloads/  (click macOS installer)"
  echo "Then run this script again."
  exit 1
fi

echo "Python found: $(python3 --version)"

if [ ! -d ".venv" ]; then
  echo "Creating .venv (first run only)..."
  python3 -m venv .venv
fi

echo "Turning on .venv..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing requirements (first run takes 1-2 minutes)..."
python -m pip install -q -r requirements.txt

echo ""
echo "Starting CommentPulse — open http://localhost:8100 in your browser"
echo "Stop it any time with Ctrl + C"
echo ""
uvicorn app.main:app --port 8100
