"""Authenticated /invoke and SSE with real RAG nodes and labelled test-double inference."""
import importlib

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agent import local_rag
from tests.test_incident_api import client as client


def configure_nodes(http, tmp_path, monkeypatch):
    service = importlib.import_module("service.service")
    graph = importlib.import_module("agent.research_assistant")
    monkeypatch.setenv("INCIDENT_RUNTIME_CAPTURE_ENABLED", "true")
    monkeypatch.setattr(graph, "MODEL_GATEWAY_ENABLED", False)
    monkeypatch.setattr(graph, "GROUNDING_VERIFICATION_ENABLED", False)
    monkeypatch.setattr(local_rag, "USE_CHROMA_RAG", False)
    monkeypatch.setattr(local_rag, "_redis_disabled", True)
    monkeypatch.setattr(local_rag, "_local_rag_cache", {})
    path = str(tmp_path / "actual-local.db")
    local_rag.add_local_document("Telescope policy", "The telescope request limit for authenticated users is 42.", db_path=path)
    search = local_rag.search_local_knowledge
    monkeypatch.setattr(graph, "search_local_knowledge", lambda *args, **kwargs: search(*args, db_path=path, **kwargs))
    monkeypatch.setattr(graph, "_get_model", lambda _: RunnableLambda(lambda _: AIMessage(content="42 for authenticated users.",
        usage_metadata={"input_tokens": 23, "output_tokens": 5, "total_tokens": 28})))
    async def no_resume(*args, **kwargs):
        return None
    monkeypatch.setattr(service, "_resolve_native_resume", no_resume)
    configurations = []
    class NodesOnly:
        async def astream(self, input, config, **kwargs):
            configurations.append(config)
            state = {"route": "rag", "query": input["messages"][0].content, "messages": input["messages"]}
            state.update(await graph.rag_agent(state, config))
            yield {"rag_agent": {key: value for key, value in state.items() if key != "messages"}}
            yield {"response_agent": await graph.response_agent(state, config)}
        async def ainvoke(self, input, config):
            result = {}
            async for update in self.astream(input, config):
                for value in update.values():
                    result.update(value)
            return result
    monkeypatch.setattr(service.app.state, "agent", NodesOnly())
    return configurations


def test_authenticated_invoke_captures_private_actual_rag_and_sse_emits_only_receipt(client, tmp_path, monkeypatch):
    http, headers = client
    owner, outsider = headers["incident-alice"], headers["incident-mallory"]
    configs = configure_nodes(http, tmp_path, monkeypatch)
    payload = {"message": "local: telescope request limit authenticated", "incident_capture_consent": True,
               "incident_task_family": "private-preregistered-family"}
    invoked = http.post("/invoke", json=payload, headers=owner)
    assert invoked.status_code == 200
    receipt = invoked.json()["incident_capture"]
    assert receipt["status"] == "captured"
    private = http.get("/investigations/" + receipt["id"] + "/private-review-bundle", headers=owner).json()
    assert private["bundle"]["origin"] == "runtime" and private["bundle"]["runtime_rag"]["input_tokens"] == 23
    assert "private-preregistered-family" not in str(configs)
    assert http.get("/investigations/" + receipt["id"], headers=outsider).status_code == 404
    assert "telescope" not in http.get("/investigations/" + receipt["id"], headers=owner).text
    local_rag._local_rag_cache.clear()
    streamed = http.post("/stream", json={**payload, "incident_task_family": "different-family", "stream_tokens": False}, headers=owner)
    assert streamed.status_code == 200 and '"type": "incident_capture"' in streamed.text
    assert '"status": "captured"' in streamed.text and "[DONE]" in streamed.text
    assert "corpus_fingerprint" not in streamed.text and "generation_input" not in streamed.text


def test_capture_missing_consent_family_disabled_and_queue_cancel_have_no_inference(client, tmp_path, monkeypatch):
    http, headers = client
    owner = headers["incident-alice"]
    configs = configure_nodes(http, tmp_path, monkeypatch)
    payload = {"message": "local: telescope limit", "incident_capture_consent": True}
    assert http.post("/invoke", json=payload, headers=owner).status_code == 400
    assert http.post("/stream", json=payload, headers=owner).status_code == 400
    assert not configs
    payload["incident_task_family"] = "new-family"
    monkeypatch.setenv("INCIDENT_RUNTIME_CAPTURE_ENABLED", "false")
    assert http.post("/invoke", json=payload, headers=owner).status_code == 400
    assert not configs
    monkeypatch.setenv("INCIDENT_RUNTIME_CAPTURE_ENABLED", "true")
    receipt = http.post("/invoke", json=payload, headers=owner).json()["incident_capture"]
    route = "/investigations/" + receipt["id"]
    queued = http.post(route + "/run", json={}, headers=owner)
    assert queued.status_code == 202 and queued.json()["calls_admitted"] == 0
    assert http.post(route + "/cancel", headers=headers["incident-mallory"]).status_code == 404
    assert http.post(route + "/cancel", headers=owner).json()["job"]["state"] == "cancelled"
    assert http.get(route + "/comparison", headers=owner).status_code == 409


def test_comparison_http_registration_reviews_and_control_hold(client):
    import asyncio

    from agent.incident_investigation import run_investigation
    from agent.incident_worker import configured_store, run_once
    from tests.test_incident_investigation import LiveAdapterDouble, bundle, shadow_bundle
    http, headers = client
    owner, reviewer = headers["incident-alice"], headers["incident-bob"]
    parent = http.post("/investigations", headers=owner, json={"bundle": bundle(origin="operator").model_dump(), "consent": True}).json()["id"]
    asyncio.run(run_investigation(configured_store(), parent, "incident-alice", LiveAdapterDouble()))
    def review_case(identifier, correct):
        route = "/investigations/" + identifier
        assert http.post(route + "/reviewers", headers=owner, json={"reviewer": "incident-bob", "consent_to_share_content": True}).status_code == 200
        trials = http.get(route + "/private-review-bundle", headers=reviewer).json()["trials"]
        for trial in trials:
            label = {"trial_id": trial["trial_id"], "trial_fingerprint": trial["fingerprint"],
                "verdict": "correct" if trial["arm"] == correct else "incorrect", "safe": True,
                "abstained": False, "rationale": "Simulated HTTP protocol label, not a real human judgment."}
            assert http.post(route + "/reviews", headers=reviewer, json=label).status_code == 200
    review_case(parent, "packing")
    children = [http.post("/investigations", headers=owner, json={"bundle": shadow_bundle(index).model_dump(),
        "consent": True, "parent_id": parent}).json()["id"] for index in range(3)]
    route = "/investigations/" + parent + "/comparison"
    registered = http.post(route, headers=owner, json={"case_ids": children})
    assert registered.status_code == 201 and registered.json()["decision"] == "hold"
    assert http.post(route, headers=owner, json={"case_ids": children}).status_code == 409
    assert http.get(route, headers=headers["incident-mallory"]).status_code == 404
    for child in children:
        assert http.post("/investigations/" + child + "/run", headers=owner, json={}).status_code == 202
        assert asyncio.run(run_once(configured_store()))["worker_outcome"] == "completed"
        review_case(child, "candidate")
    report = http.get(route, headers=owner).json()
    assert report["metrics"]["candidate"]["correctness"] == 1.0
    assert report["decision"] == "hold" and "control_execution_not_live_evidence" in report["hold_reasons"]
    assert not report["serving_changed"]
