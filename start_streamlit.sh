#!/bin/bash
streamlit run app/streamlit_app.py \
    --server.port 10000 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false