"""
WSGI entrypoint for Render and Gunicorn web servers.
"""
from backend.app import app

if __name__ == "__main__":
    app.run()
