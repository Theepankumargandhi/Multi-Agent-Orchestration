from langchain_core.messages import AIMessage
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from uuid import uuid4

from service import app
from schema import ChatMessage, model_validate_compat

client = TestClient(app)


def _auth_headers(c: TestClient, user_id: str | None = None, password: str = "Password123!"):
    user_id = user_id or f"test-user-{uuid4().hex[:8]}"
    register = c.post("/auth/register", json={"user_id": user_id, "password": password})
    if register.status_code == 409:
        login = c.post("/auth/login", json={"user_id": user_id, "password": password})
        assert login.status_code == 200
        token = login.json()["access_token"]
    else:
        assert register.status_code == 200
        token = register.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_auth_register_and_login():
    user_id = f"login-user-{uuid4().hex[:8]}"
    password = "Password123!"
    with client as c:
        register = c.post("/auth/register", json={"user_id": user_id, "password": password})
        assert register.status_code == 200
        assert register.json()["user_id"] == user_id
        assert register.json()["access_token"]

        login = c.post("/auth/login", json={"user_id": user_id, "password": password})
        assert login.status_code == 200
        assert login.json()["user_id"] == user_id
        assert login.json()["access_token"]


def test_invoke():
    agent = type("MockAgent", (), {})()
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    agent_response = {"messages": [AIMessage(content=ANSWER)]}
    agent.ainvoke = AsyncMock(return_value=agent_response)

    with client as c:
        c.app.state.agent = agent
        headers = _auth_headers(c)
        response = c.post("/invoke", json={"message": QUESTION}, headers=headers)
        assert response.status_code == 200

    agent.ainvoke.assert_awaited_once()
    input_message = agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = model_validate_compat(ChatMessage, response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


@patch("service.service.LangsmithClient")
def test_feedback(mock_client):
    ls_instance = mock_client.return_value
    ls_instance.create_feedback.return_value = None
    body = {"run_id": "847c6285-8fc9-4560-a83f-4e6285809254", "key": "human-feedback-stars", "score": 0.8}
    with client as c:
        headers = _auth_headers(c)
        response = c.post("/feedback", json=body, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    ls_instance.create_feedback.assert_called_once_with(
        run_id="847c6285-8fc9-4560-a83f-4e6285809254",
        key="human-feedback-stars",
        score=0.8,
    )


def test_store_is_isolated_by_user():
    agent = type("MockAgent", (), {})()
    agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="ok")]})
    shared_thread = "shared-thread-id"

    with client as c:
        c.app.state.agent = agent
        headers_a = _auth_headers(c, user_id=f"user-a-{uuid4().hex[:6]}")
        headers_b = _auth_headers(c, user_id=f"user-b-{uuid4().hex[:6]}")

        resp_a = c.post(
            "/invoke",
            json={"message": "message from A", "thread_id": shared_thread},
            headers=headers_a,
        )
        assert resp_a.status_code == 200

        resp_b = c.post(
            "/invoke",
            json={"message": "message from B", "thread_id": shared_thread},
            headers=headers_b,
        )
        assert resp_b.status_code == 200

        store_a = c.get(f"/store/{shared_thread}", headers=headers_a)
        store_b = c.get(f"/store/{shared_thread}", headers=headers_b)
        assert store_a.status_code == 200
        assert store_b.status_code == 200

    contents_a = [m["content"] for m in store_a.json()["messages"]]
    contents_b = [m["content"] for m in store_b.json()["messages"]]
    assert any("message from A" in content for content in contents_a)
    assert not any("message from B" in content for content in contents_a)
    assert any("message from B" in content for content in contents_b)
    assert not any("message from A" in content for content in contents_b)


def test_store_threads_are_isolated_by_user():
    agent = type("MockAgent", (), {})()
    agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="ok")]})

    with client as c:
        c.app.state.agent = agent
        headers_a = _auth_headers(c, user_id=f"user-a-{uuid4().hex[:6]}")
        headers_b = _auth_headers(c, user_id=f"user-b-{uuid4().hex[:6]}")

        c.post("/invoke", json={"message": "A-1", "thread_id": "thread-a-1"}, headers=headers_a)
        c.post("/invoke", json={"message": "A-2", "thread_id": "thread-a-2"}, headers=headers_a)
        c.post("/invoke", json={"message": "B-1", "thread_id": "thread-b-1"}, headers=headers_b)

        threads_a = c.get("/store/threads", headers=headers_a)
        threads_b = c.get("/store/threads", headers=headers_b)
        assert threads_a.status_code == 200
        assert threads_b.status_code == 200

    ids_a = {t["thread_id"] for t in threads_a.json()["threads"]}
    ids_b = {t["thread_id"] for t in threads_b.json()["threads"]}
    assert "thread-a-1" in ids_a
    assert "thread-a-2" in ids_a
    assert "thread-b-1" not in ids_a
    assert "thread-b-1" in ids_b


@patch("service.service.perform_web_search")
def test_web_search_preview_endpoint(mock_web_search):
    mock_web_search.return_value = (
        "- Example AI News\n"
        "  Link: https://example.com/ai-news\n"
        "  Date: 2026-03-01\n"
        "  Snippet: Example summary.",
        {"cache_hit": False, "source": "web_live"},
    )

    with client as c:
        headers = _auth_headers(c)
        resp = c.post(
            "/web_search/preview",
            json={"query": "latest ai news last week", "recency_days": 7, "max_results": 5},
            headers=headers,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "latest ai news last week"
    assert body["recency_days"] == 7
    assert body["count"] >= 1
    assert body["items"][0]["title"] == "Example AI News"
    assert body["items"][0]["url"] == "https://example.com/ai-news"


def test_monitoring_endpoints_are_public():
    with client as c:
        health = c.get("/healthz")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        ready = c.get("/readyz")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ready"}

        metrics = c.get("/metrics")
        assert metrics.status_code == 200
        assert "http_requests_total" in metrics.text


def test_hitl_web_decision_is_recorded_and_listed():
    with client as c:
        headers = _auth_headers(c, user_id=f"hitl-user-{uuid4().hex[:6]}")
        thread_id = f"hitl-thread-{uuid4().hex[:6]}"
        query = "latest ai news last week"

        record = c.post(
            "/hitl/web_decision",
            json={
                "query": query,
                "decision": "rejected",
                "thread_id": thread_id,
                "reason": "results are outside last week",
                "recency_days": 7,
                "preview_count": 5,
                "all_within_recency": False,
                "source": "web_live",
                "cache_hit": False,
            },
            headers=headers,
        )
        assert record.status_code == 200
        assert record.json()["status"] == "recorded"

        listed = c.get(
            "/hitl/web_decisions",
            params={"limit": 20, "thread_id": thread_id},
            headers=headers,
        )
        assert listed.status_code == 200
        body = listed.json()
        assert body["count"] >= 1
        assert any(
            event["query"] == query
            and event["decision"] == "rejected"
            and event["reason"] == "results are outside last week"
            for event in body["events"]
        )
