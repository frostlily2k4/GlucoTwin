#!/usr/bin/env bash
# Rebuild everything from scratch, then open the dashboard
set -e
python src/generate_data.py
python src/train.py
python docs/make_architecture.py
streamlit run app.py
