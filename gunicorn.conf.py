"""
Gunicorn configuration for Render deployment.
Automatically binds to 0.0.0.0:$PORT and keeps memory usage within free-tier limits.
"""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = 1
threads = 4
timeout = 120
