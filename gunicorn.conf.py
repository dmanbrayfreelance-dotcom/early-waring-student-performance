"""
Gunicorn configuration for Render deployment.
Automatically binds to 0.0.0.0:$PORT and keeps memory usage within free-tier limits.
"""
import os

port = os.environ.get("PORT") or "10000"
bind = f"0.0.0.0:{port}"
workers = 1
threads = 4
timeout = 120
