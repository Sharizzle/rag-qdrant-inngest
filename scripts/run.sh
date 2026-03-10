#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

docker run -d --rm -p 6333:6333 -p 6334:6334 qdrant/qdrant
uv run uvicorn main:app --reload &
inngest dev -u http://127.0.0.1:8000/api/inngest &
uv run streamlit run streamlit_app.py &

wait