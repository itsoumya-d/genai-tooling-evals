#!/usr/bin/env bash
set -euo pipefail

echo "Running smoke test against GenAI Tooling Evals service..."
uvicorn genai_evals.app:app --host 127.0.0.1 --port 8000 &
PID=$!
trap "kill -9 $PID 2>/dev/null || true" EXIT

for i in {1..30}; do
  if curl -sf http://127.0.0.1:8000/health | grep -q '"status":"ok"'; then
    echo "Smoke test PASSED: /health responded ok"
    exit 0
  fi
  sleep 0.5
done

echo "Smoke test FAILED: server did not start in time"
exit 1
