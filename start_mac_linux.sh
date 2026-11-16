#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
python3 -m streamlit run credit_analytics_bot.py
