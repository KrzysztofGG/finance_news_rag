#!/bin/bash
# Start the Finance RAG Agent API server

cd "$(dirname "$0")/.."
uvicorn agent.api:app --host 0.0.0.0 --port 8000 --reload
