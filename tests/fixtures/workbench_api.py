"""Authored browser-QA server. Synthetic records only; never imports an agent or calls a provider.

python -m uvicorn workbench_api:app --app-dir tests/fixtures --host 127.0.0.1 --port 8012
"""

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="AUTHORED Workbench UI fixture — not a live agent")
TASK_ID = "00000000-0000-4000-8000-000000000001"
state = {"submitted": False, "status": "awaiting_approval", "request": {}, "messages": [], "stream_requests": 0}


@app.middleware("http")
async def owner(request, call_next):
    if request.url.path not in {"/capabilities", "/auth/login", "/healthz"}:
        if request.headers.get("authorization") != "Bearer authored-browser-session":
            return JSONResponse(status_code=401, content={"detail": "fixture_unauthorized"})
    return await call_next(request)


@app.get("/healthz")
def health():
    return {"fixture": True, "provider_calls": 0, "sandbox_executions": 0}


@app.get("/capabilities")
def capabilities():
    return {"models": [{"id": "authored-offline-model", "label": "Authored offline fixture (no provider)"}],
            "features": {"user_authentication": True, "streaming": True, "sandboxed_code_agent": True}}


@app.post("/auth/login")
async def login(request: Request):
    payload = await request.json()
    if payload != {"user_id": "reviewer", "password": "authored-password"}:
        raise HTTPException(status_code=401)
    return {"user_id": "reviewer", "access_token": "authored-browser-session", "expires_in": 3600}


@app.get("/workbench/config")
def config():
    return {"version": "0.1.0", "identity": "reviewer", "code_enabled": True,
            "evidence_mode": "authored_fixture", "repositories": ["authored-example"],
            "coding_models": ["authored-offline-model"], "workflow": "verified_pr (authored fixture)",
            "test_command": ["python", "-m", "pytest", "-q"], "integrations": {}}


@app.get("/store/threads")
def threads():
    return {"threads": [{"thread_id": "authored-thread", "last_message_preview": "Authored browser check", "message_count": len(state["messages"])}] if state["messages"] else []}


@app.get("/store/{thread_id}")
def history(thread_id: str):
    return {"messages": state["messages"][::-1]}


@app.post("/stream")
async def stream(request: Request):
    payload = await request.json()
    state["stream_requests"] += 1
    question = payload["message"]
    answer = "Authored UI fixture: Atlas p95 latency is 180 ms. [atlas-report.pdf, page 2]\n\nThis verifies browser rendering and transport—not a live model answer."
    state["messages"].extend([{"role": "human", "content": question}, {"role": "ai", "content": answer}])

    async def events():
        records = [{"type": "message", "content": {"type": "ai", "content": "", "tool_calls": [{"id": "authored-tool", "name": "authored_local_retrieval", "args": {"query": question}}]}},
                   {"type": "message", "content": {"type": "tool", "content": "Synthetic evidence: Atlas p95 latency is 180 ms.", "tool_call_id": "authored-tool"}},
                   {"type": "token", "content": "Authored UI fixture: "},
                   {"type": "message", "content": {"type": "ai", "content": answer, "run_id": "authored-run"}}]
        for record in records:
            yield "data: " + json.dumps(record) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def record():
    return {"task_id": TASK_ID, "status": state["status"], "created_at": "authored timestamp", "updated_at": "authored timestamp",
            "request": state["request"], "result": {"summary": "Authored fixture: changed the boundary condition and retained the owner tests. No actual repair was executed.",
            "changed_files": ["example.py"], "model_calls": 0, "tool_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "usage_accounting_complete": None, "verification": {"final_decision": "verified", "blocking_reasons": []},
            "baseline_test": {"exit_code": 1, "stdout": "Authored fixture: expected boundary failure", "stderr": ""},
            "final_test": {"exit_code": 0, "stdout": "Authored fixture: 1 passed (not executed)", "stderr": ""},
            "observations": [{"iteration": 1, "action": "test", "stage": "authored-fixture", "ok": True, "summary": "Synthetic test observation; no sandbox execution."}]}}


@app.get("/workbench/code/tasks")
def tasks():
    return {"tasks": [{"task_id": TASK_ID, "status": state["status"], "repository": "authored-example", "issue_preview": state["request"]["issue"], "model": "authored-offline-model", "updated_at": "authored timestamp"}] if state["submitted"] else []}


@app.post("/workbench/code/tasks", status_code=202)
async def submit(request: Request):
    state["request"] = await request.json()
    state["submitted"], state["status"] = True, "awaiting_approval"
    return record()


@app.get("/code/tasks/{task_id}")
def task(task_id: str):
    if task_id != TASK_ID or not state["submitted"]:
        raise HTTPException(status_code=404)
    return record()


@app.get("/code/tasks/{task_id}/events")
def task_events(task_id: str):
    return {"events": [{"occurred_at": "authored timestamp", "from_status": "queued", "to_status": state["status"], "actor": "authored-fixture"}]}


@app.post("/code/tasks/{task_id}/decision")
async def decide(task_id: str, request: Request):
    payload = await request.json()
    if state["status"] != "awaiting_approval":
        raise HTTPException(status_code=409)
    state["status"] = "approved" if payload["approve"] else "rejected"
    return record()


@app.get("/code/tasks/{task_id}/patch")
def patch(task_id: str):
    if state["status"] != "approved":
        raise HTTPException(status_code=403)
    return {"patch": "--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-return value < 0\n+return value <= 0\n"}


@app.get("/code/tasks/{task_id}/dossier")
def dossier(task_id: str):
    return {"evidence_kind": "authored_browser_fixture", "provider_calls": 0, "sandbox_executions": 0, "task_id": task_id}
