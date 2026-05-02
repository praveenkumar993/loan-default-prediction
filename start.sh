#!/bin/bash

# Start FastAPI in background
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in foreground
streamlit run app/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false