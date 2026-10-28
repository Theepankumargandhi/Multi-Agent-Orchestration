import os

import uvicorn
from dotenv import load_dotenv

# Agent modules read their settings when imported, so .env must be loaded before the app.
load_dotenv()

from service import app  # noqa: E402

host = os.getenv("HOST", "0.0.0.0")
port = int(os.getenv("PORT", "8000"))
uvicorn.run(app, host=host, port=port)
