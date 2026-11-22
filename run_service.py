import os

import uvicorn
from dotenv import load_dotenv

# Agent modules read their settings when imported, so .env must be loaded before the app.
load_dotenv()

from service import app  # noqa: E402

host = os.getenv("HOST", "0.0.0.0")
port = int(os.getenv("PORT", "8000"))
# WEB_DIR serves the built UI from this port too (the API moves under /api), for one-container hosts.
web_dir = os.getenv("WEB_DIR", "").strip()
if web_dir:
    from service.demo import SinglePortApp

    uvicorn.run(SinglePortApp(app, web_dir), host=host, port=port)
else:
    uvicorn.run(app, host=host, port=port)
