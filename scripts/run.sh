#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

docker run -d --rm -p 6333:6333 -p 6334:6334 qdrant/qdrant
uv run uvicorn main:app --reload &
inngest dev -u http://127.0.0.1:8000/api/inngest &
uv run streamlit run streamlit_app.py &

# Allow a moment for servers to start before opening browser
sleep 2

# Try to open in browser (cross-platform, best effort)
open_cmd() {
  if command -v xdg-open &> /dev/null; then
    xdg-open "$1"
  elif command -v open &> /dev/null; then
    open "$1"
  elif command -v start &> /dev/null; then
    start "$1"
  else
    echo "Please visit: $1"
  fi
}

open_cmd "http://localhost:8288/runs"
open_cmd "http://localhost:8501"

wait