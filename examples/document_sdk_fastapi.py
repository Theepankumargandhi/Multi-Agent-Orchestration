"""Local integration example, NOT an authenticated multi-tenant deployment.

Run with: python -m uvicorn examples.document_sdk_fastapi:app --host 127.0.0.1 --port 8011
"""
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from agentforge_sdk import AnswerResult, DocumentRAG, SDKError
from agentforge_sdk.demo import demo_pages


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=1, ge=1, le=20)


def create_app(rag: DocumentRAG):
    """Caller must scope the injected corpus to its authorized readers."""
    app = FastAPI(title="AgentForge text-evidence SDK example", debug=False)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, _exception):
        # Default validation payloads can echo submitted private question text.
        return JSONResponse(status_code=422, content={"detail": "invalid_request"})

    @app.post("/answer", response_model=AnswerResult)
    async def answer(request: Question):
        try:
            return await rag.aanswer(request.question, top_k=request.top_k)
        except (ValueError, SDKError):
            return JSONResponse(status_code=400, content={"detail": "request_or_backend_rejected"})

    return app


app = create_app(DocumentRAG(demo_pages()))
