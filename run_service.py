import os

from dotenv import load_dotenv
import uvicorn

from service import app

load_dotenv()
host = os.getenv("HOST", "0.0.0.0")
port = int(os.getenv("PORT", "8000"))
uvicorn.run(app, host=host, port=port)
