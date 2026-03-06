from datetime import datetime, timedelta, timezone
import base64
import os
import re
from urllib.parse import parse_qs

import numexpr
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, StateGraph, MessagesState

from agent.tools import perform_web_search
from agent.local_rag import init_local_knowledge_store, search_local_knowledge
from agent.knowledge_graph import query_knowledge_graph
from agent.llama_guard import llama_guard, LlamaGuardOutput, SafetyAssessment
from agent.mcp_client import mcp_web_search, mcp_calculator


class AgentState(MessagesState):
    safety: LlamaGuardOutput
    route: str
    route_confidence: float
    route_reason: str
    query: str
    rewritten_query: str
    rewrite_done: bool
    rewrite_notes: str
    recency_query: str
    recency_days: int
    recency_notes: str
    web_hitl_decision: str
    web_hitl_reject_reason: str
    web_hitl_pending_query: str
    web_hitl_pending_route: str
    web_hitl_pending_recency_query: str
    web_hitl_pending_recency_days: int
    web_hitl_pending_web_notes: str
    web_hitl_pending_source_meta: str
    web_hitl_preview_count: int
    web_hitl_all_within_recency: bool
    web_hitl_audit_query: str
    web_notes: str
    web_source_meta: str
    rag_notes: str
    rag_source_meta: str
    kg_notes: str
    kg_source_meta: str
    math_result: str
    final_response: str
    answer_source_meta: str
    evaluation_score: int
    evaluation_report: str


# 12 specialized agents in this orchestration graph.
SPECIALIZED_AGENTS = [
    "safety_agent",
    "intent_router_agent",
    "clarification_agent",
    "query_rewriter_agent",
    "recency_guard_agent",
    "web_hitl_gate_agent",
    "web_search_agent",
    "knowledge_graph_agent",
    "rag_agent",
    "math_agent",
    "response_agent",
    "evaluation_agent",
]

# NOTE: models with streaming=True will send tokens as they are generated
# if the /stream endpoint is called with stream_tokens=True (the default)
_model_cache = {}
DEFAULT_VAGUE_NEWS_TOPIC = os.getenv("DEFAULT_VAGUE_NEWS_TOPIC", "ai").strip().lower()
HYBRID_ROUTER_ENABLE = os.getenv("HYBRID_ROUTER_ENABLE", "true").strip().lower() in {"1", "true", "yes", "on"}
HYBRID_ROUTER_MIN_CONFIDENCE = float(os.getenv("HYBRID_ROUTER_MIN_CONFIDENCE", "0.75"))
GRAPH_WEB_HITL_ENABLED = os.getenv("GRAPH_WEB_HITL_ENABLED", os.getenv("WEB_HITL_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
GRAPH_WEB_HITL_MAX_RESULTS = max(1, min(int(os.getenv("GRAPH_WEB_HITL_MAX_RESULTS", os.getenv("WEB_HITL_MAX_RESULTS", "5"))), 10))


def _build_model(model_name: str) -> BaseChatModel:
    if model_name == "llama-3.1-70b":
        return ChatGroq(model="llama-3.1-70b-versatile", temperature=0.2)
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)


current_date = datetime.now().strftime("%B %d, %Y")
base_instructions = f"""
You are the final response assistant in a multi-agent orchestration system.
Today's date is {current_date}.

Rules:
- Be concise, correct, and helpful.
- If web evidence is used, include 1-3 markdown citations.
- If local RAG evidence is used, reference the local source labels.
- If knowledge-graph evidence is used, explain the relationship path clearly.
- For math results, show human-readable equations (e.g., 300 * 200).
""".strip()


def _get_model(config: RunnableConfig) -> BaseChatModel:
    model_name = config["configurable"].get("model", "gpt-4o-mini")
    # Fallback if requested provider key is missing in env.
    if model_name == "gpt-4o-mini" and not os.getenv("OPENAI_API_KEY") and os.getenv("GROQ_API_KEY"):
        model_name = "llama-3.1-70b"
    if model_name == "llama-3.1-70b" and not os.getenv("GROQ_API_KEY") and os.getenv("OPENAI_API_KEY"):
        model_name = "gpt-4o-mini"

    if model_name not in _model_cache:
        _model_cache[model_name] = _build_model(model_name)
    return _model_cache[model_name]


async def _call_llm(system: str, user: str, config: RunnableConfig) -> str:
    model = _get_model(config)
    runnable = RunnableLambda(lambda _: [SystemMessage(content=system), ("human", user)]) | model
    response = await runnable.ainvoke({}, config)
    return (response.content or "").strip()


def _latest_user_query(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            return msg.content.strip()
    return ""


def _active_query(state: AgentState) -> str:
    return (state.get("rewritten_query") or state.get("query") or "").strip()


def _web_query(state: AgentState) -> str:
    return (state.get("recency_query") or _active_query(state)).strip()


def _has_local_prefix(query: str) -> bool:
    q = (query or "").strip().lower()
    return bool(re.match(r"^local\s*:", q))


def _strip_local_prefix(query: str) -> str:
    q = (query or "").strip()
    return re.sub(r"(?i)^local\s*:\s*", "", q, count=1).strip()


def _looks_like_math(query: str) -> bool:
    q = (query or "").lower()
    if any(k in q for k in ["calculate", "math", "equation", "solve"]):
        return True
    return bool(re.search(r"[\d\s\+\-\*/\(\)\.\^]{3,}", q))


def _extract_expression(query: str) -> str:
    cleaned = re.sub(r"[^0-9\+\-\*/\(\)\.\s\^]", " ", query or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _looks_like_relation_query(query: str) -> bool:
    q = (query or "").lower()
    relation_hints = [
        "relationship",
        "related",
        "relation",
        "connect",
        "connected",
        "connection",
        "dependency",
        "depends on",
        "impact of",
        "influence",
        "how does",
        "how is",
        "difference between",
        "compare",
    ]
    return any(hint in q for hint in relation_hints)


def _has_any_phrase(text: str, phrases: list[str]) -> bool:
    t = (text or "").lower()
    return any(p in t for p in phrases)


def _looks_like_greeting(query: str) -> bool:
    q = (query or "").strip().lower()
    if q in {"hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening"}:
        return True
    return any(q.startswith(prefix) for prefix in ("hi ", "hello ", "hey "))


def _is_vague_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return True

    tokens = re.findall(r"[a-z0-9]+", q)
    if len(tokens) <= 2 and not _looks_like_math(q) and not _looks_like_greeting(q):
        return True

    vague_phrases = {
        "tell me more",
        "more details",
        "latest news",
        "latest updates",
        "news update",
        "update me",
        "help me",
        "explain more",
        "what about this",
        "what about that",
    }
    if q in vague_phrases:
        return True

    pronoun_only = {"it", "this", "that", "they", "them", "something", "anything"}
    if len(tokens) <= 4 and any(t in pronoun_only for t in tokens):
        return True

    return False


def _needs_clarification(query: str) -> bool:
    """
    True when user intent cannot be safely disambiguated by rewrite alone.
    """
    q = (query or "").strip().lower()
    if not q:
        return True

    tokens = re.findall(r"[a-z0-9]+", q)
    pronoun_only = {"it", "this", "that", "they", "them", "something", "anything"}
    if len(tokens) <= 4 and any(t in pronoun_only for t in tokens):
        return True

    # Help requests without clear objective are better handled by clarification.
    if q in {"help", "help me", "i need help", "can you help"}:
        return True

    return False


def _rule_based_rewrite(query: str) -> str:
    """
    Deterministic rewrite rules for common vague prompts.

    If DEFAULT_VAGUE_NEWS_TOPIC is empty/none, this behavior is disabled.
    """
    q = (query or "").strip().lower()
    if not q:
        return ""

    if DEFAULT_VAGUE_NEWS_TOPIC in {"", "none", "off", "false"}:
        return ""

    news_like = {
        "news",
        "news update",
        "latest news",
        "latest updates",
        "latest update",
        "updates",
        "update me",
    }
    if q in news_like:
        topic = DEFAULT_VAGUE_NEWS_TOPIC
        return f"latest {topic} news today with reliable sources"

    return ""


def _extract_recency_days(query: str) -> int | None:
    q = (query or "").lower()
    if "today" in q:
        return 1
    if "yesterday" in q:
        return 2
    if any(k in q for k in ["this week", "weekly", "past week", "last week"]):
        return 7
    if any(k in q for k in ["this month", "monthly", "past month", "last month"]):
        return 30
    return None


def _parse_ymd_date(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _b64url_decode_text(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    try:
        padding = "=" * ((4 - len(text) % 4) % 4)
        raw = base64.urlsafe_b64decode(text + padding)
        return raw.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _parse_web_hitl_control_message(user_text: str) -> dict[str, str | int] | None:
    """
    Parse Streamlit button control payload:
    __WEB_HITL__|<action>|<route>|<recency_days>|<query_b64>|<reason_b64>
    """
    text = (user_text or "").strip()
    # Preferred format:
    # WEB_HITL_DECISION?action=<approve|reject>&route=<web|hybrid>&days=<int>&query_b64=<...>&reason_b64=<...>
    marker_new = "WEB_HITL_DECISION?"
    idx_new = text.find(marker_new)
    if idx_new >= 0:
        payload = text[idx_new + len(marker_new):]
        params = parse_qs(payload, keep_blank_values=True)
        action = str((params.get("action") or [""])[0]).strip().lower()
        if action not in {"approve", "reject"}:
            return None
        route = str((params.get("route") or ["web"])[0]).strip().lower()
        if route not in {"web", "hybrid"}:
            route = "web"
        try:
            recency_days = max(0, min(int((params.get("days") or ["0"])[0]), 365))
        except Exception:
            recency_days = 0
        query = _b64url_decode_text(str((params.get("query_b64") or [""])[0]))
        reason = _b64url_decode_text(str((params.get("reason_b64") or [""])[0]))
        return {
            "action": action,
            "route": route,
            "recency_days": recency_days,
            "query": query,
            "reason": reason,
        }

    # Backward-compatible legacy format:
    # __WEB_HITL__|<action>|<route>|<recency_days>|<query_b64>|<reason_b64>
    marker_old = "__WEB_HITL__|"
    idx_old = text.find(marker_old)
    if idx_old >= 0:
        payload = text[idx_old:]
        parts = payload.split("|", 5)
        if len(parts) != 6:
            return None
        _, action_raw, route_raw, days_raw, query_b64, reason_b64 = parts
        action = action_raw.strip().lower()
        if action not in {"approve", "reject"}:
            return None
        route = route_raw.strip().lower()
        if route not in {"web", "hybrid"}:
            route = "web"
        try:
            recency_days = max(0, min(int(days_raw or 0), 365))
        except Exception:
            recency_days = 0
        query = _b64url_decode_text(query_b64)
        reason = _b64url_decode_text(reason_b64)
        return {
            "action": action,
            "route": route,
            "recency_days": recency_days,
            "query": query,
            "reason": reason,
        }

    return None


def _looks_like_web_hitl_candidate(query: str, recency_days: int) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if recency_days > 0:
        return True
    recency_terms = ("latest", "recent", "current", "today", "this week", "last week", "past")
    web_terms = ("news", "update", "updates", "headline", "headlines", "happening", "trend", "trends")
    return any(term in q for term in recency_terms) and any(term in q for term in web_terms)


def _parse_hitl_decision(user_text: str) -> tuple[str, str]:
    text = (user_text or "").strip()
    lower = text.lower()

    if not text:
        return "", ""

    if lower.startswith("approve"):
        return "approved", ""
    if lower in {"yes", "y", "ok", "okay", "continue", "proceed"}:
        return "approved", ""

    if lower.startswith("reject"):
        reason = text[len("reject"):].strip(" :-")
        return "rejected", reason
    if lower.startswith("no"):
        reason = text[len("no"):].strip(" :-")
        return "rejected", reason

    return "", ""


def _format_web_hitl_preview(
    query: str,
    recency_days: int,
    entries: list[dict[str, str]],
    all_within_recency: bool,
) -> str:
    lines = [
        "Human approval required before web-answer generation.",
        "",
        f"Query: {query}",
    ]
    if recency_days > 0:
        lines.append(f"Recency target: last {recency_days} days")
    lines.append("")

    if all_within_recency:
        lines.append("All dated preview results are within the recency window.")
    else:
        lines.append("Some results may be outside your recency window.")
    lines.append("")

    if entries:
        lines.append("Preview results:")
        for i, entry in enumerate(entries[:GRAPH_WEB_HITL_MAX_RESULTS], start=1):
            title = entry.get("title", "").strip() or "Untitled"
            url = entry.get("url", "").strip()
            date = entry.get("date", "").strip() or "date unknown"
            snippet = entry.get("snippet", "").strip()
            if url:
                lines.append(f"{i}. [{title}]({url}) ({date})")
            else:
                lines.append(f"{i}. {title} ({date})")
            if snippet:
                lines.append(f"   - {snippet}")
    else:
        lines.append("No preview results available.")

    lines.extend(
        [
            "",
            "Reply with `approve` to continue, or `reject: <reason>` to stop.",
        ]
    )
    return "\n".join(lines).strip()


def _parse_route_classifier_output(text: str) -> tuple[str, float, str]:
    """
    Parse classifier output into (route, confidence, reason).

    Expected flexible formats, e.g.:
    route: web
    confidence: 0.84
    reason: latest/current intent
    """
    raw = (text or "").strip()
    route = ""
    confidence = 0.0
    reason = ""

    for line in raw.splitlines():
        lower = line.lower().strip()
        if lower.startswith("route:"):
            route = line.split(":", 1)[1].strip().lower()
        elif lower.startswith("confidence:"):
            value = line.split(":", 1)[1].strip()
            try:
                confidence = float(value)
            except ValueError:
                confidence = 0.0
        elif lower.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    valid_routes = {"clarify", "rewrite", "math", "web", "rag", "kg", "hybrid", "general"}
    if route not in valid_routes:
        route = "general"
    confidence = max(0.0, min(1.0, confidence))
    return route, confidence, reason or "LLM route classifier fallback."


async def _llm_route_classify(query: str, config: RunnableConfig) -> tuple[str, float, str]:
    system = (
        "You classify user queries into one route for a multi-agent system.\n"
        "Allowed routes: clarify, rewrite, math, web, rag, kg, hybrid, general.\n"
        "Definitions:\n"
        "- clarify: user intent is ambiguous and requires follow-up.\n"
        "- rewrite: vague but inferable and can be rewritten (e.g., 'news update').\n"
        "- math: arithmetic/calculation/equation solving.\n"
        "- web: latest/current/news/search intent from the web.\n"
        "- rag: local project/docs/codebase/local knowledge intent.\n"
        "- kg: relationship reasoning over local knowledge (entity-to-entity links).\n"
        "- hybrid: needs both web and local/project context.\n"
        "- general: regular non-time-sensitive Q&A/chat.\n"
        "Return exactly 3 lines:\n"
        "route: <one route>\n"
        "confidence: <0.00-1.00>\n"
        "reason: <short reason>\n"
    )
    user = f"Classify this query:\n{query}"
    try:
        text = await _call_llm(system, user, config)
        return _parse_route_classifier_output(text)
    except Exception as e:
        return "general", 0.0, f"LLM router unavailable: {e}"


def _build_execution_flow(state: AgentState) -> list[str]:
    route = (state.get("route") or "general").lower()
    rewrite_done = bool(state.get("rewrite_done"))

    flow: list[str] = ["safety_agent", "intent_router_agent"]

    if rewrite_done:
        flow.extend(["query_rewriter_agent", "intent_router_agent"])

    match route:
        case "clarify":
            flow.extend(["clarification_agent", "evaluation_agent"])
        case "web":
            flow.extend(["recency_guard_agent", "web_hitl_gate_agent", "web_search_agent", "response_agent", "evaluation_agent"])
        case "kg":
            flow.extend(["knowledge_graph_agent", "rag_agent", "response_agent", "evaluation_agent"])
        case "hybrid":
            flow.extend(["recency_guard_agent", "web_hitl_gate_agent", "web_search_agent", "rag_agent", "response_agent", "evaluation_agent"])
        case "rag":
            flow.extend(["rag_agent", "response_agent", "evaluation_agent"])
        case "math":
            flow.extend(["math_agent", "response_agent", "evaluation_agent"])
        case _:
            flow.extend(["response_agent", "evaluation_agent"])

    return flow


def _finalize_user_output(text: str, state: AgentState) -> str:
    eval_marker = "_Evaluated by evaluation agent._"
    clean = (text or "").strip()
    if not clean:
        clean = eval_marker
    if eval_marker.lower() in clean.lower():
        # Already formatted.
        return clean

    flow = _build_execution_flow(state)
    unique_agents = len(set(flow))
    flow_line = " -> ".join(flow)
    meta = (
        f"{eval_marker}\n"
        f"_Agents used: {unique_agents}_\n"
        f"_Flow: {flow_line}_"
    )
    source_bits: list[str] = []
    for key in ("web_source_meta", "rag_source_meta", "kg_source_meta", "answer_source_meta"):
        val = (state.get(key) or "").strip()
        lower = val.lower()
        if "cache" in lower or "mcp" in lower or "kg" in lower:
            source_bits.append(val)
    if source_bits:
        meta += "\n" + f"_Sources: {' | '.join(source_bits)}_"
    return f"{clean}\n\n{meta}"


async def safety_agent(state: AgentState, config: RunnableConfig):
    safety_output = await llama_guard("User", state["messages"])
    latest_user = _latest_user_query(state)
    control = _parse_web_hitl_control_message(latest_user)
    query = str(control.get("query") or "").strip() if control else latest_user
    if not query:
        query = latest_user
    return {
        "safety": safety_output,
        "query": query,
        "route_confidence": 0.0,
        "route_reason": "",
        "rewritten_query": "",
        "rewrite_done": False,
        "rewrite_notes": "",
        "recency_query": "",
        "recency_days": 0,
        "recency_notes": "",
        "web_hitl_decision": "",
        "web_hitl_reject_reason": "",
        "web_hitl_audit_query": "",
        "web_source_meta": "",
        "rag_source_meta": "",
        "kg_notes": "",
        "kg_source_meta": "",
        "evaluation_score": 0,
        "evaluation_report": "",
        "answer_source_meta": "",
    }


async def intent_router_agent(state: AgentState, config: RunnableConfig):
    query = _active_query(state)
    control = _parse_web_hitl_control_message(_latest_user_query(state))
    rewrite_done = bool(state.get("rewrite_done", False))
    q = query.lower()
    forced_local = _has_local_prefix(query)
    pending_hitl_query = (state.get("web_hitl_pending_query") or "").strip()

    if control:
        control_route = str(control.get("route") or "web").strip().lower()
        if control_route not in {"web", "hybrid"}:
            control_route = "web"
        return {
            "route": control_route,
            "route_confidence": 1.0,
            "route_reason": "Explicit web HITL control message.",
        }

    # Guardrail: never treat leaked HITL control payload as a web search query.
    if "web_hitl" in q and not pending_hitl_query:
        return {
            "route": "clarify",
            "route_confidence": 1.0,
            "route_reason": "Leaked HITL control payload without pending context.",
        }

    if pending_hitl_query:
        pending_route = (state.get("web_hitl_pending_route") or "web").strip().lower()
        if pending_route not in {"web", "hybrid"}:
            pending_route = "web"
        return {
            "route": pending_route,
            "route_confidence": 1.0,
            "route_reason": "Pending web HITL decision in progress.",
        }

    web_hints = [
        "latest", "news", "today", "current", "recent", "update", "updates",
        "web", "online", "search", "headlines",
    ]
    rag_hints = [
        "this project", "this repo", "repository", "codebase", "source code",
        "service endpoint", "streamlit", "fastapi", "langgraph",
        "agent-service-toolkit", "local database", "rag",
    ]
    relation_hints = [
        "relationship",
        "related",
        "relation",
        "connect",
        "connected",
        "dependency",
        "depends on",
        "compare",
        "difference between",
        "how does",
        "how is",
    ]

    route = "general"
    route_confidence = 0.6
    route_reason = "Default general fallback."
    low_signal = False

    if forced_local:
        stripped = _strip_local_prefix(query)
        if not stripped:
            route = "clarify"
        elif _looks_like_relation_query(stripped) or _has_any_phrase(stripped.lower(), relation_hints):
            route = "kg"
        else:
            route = "rag"
        route_confidence = 1.0
        route_reason = "Forced local prefix."
    elif not rewrite_done and _needs_clarification(query):
        route = "clarify"
        route_confidence = 0.95
        route_reason = "Ambiguous/pronoun/help query requires clarification."
    elif not rewrite_done and _is_vague_query(query):
        route = "rewrite"
        route_confidence = 0.9
        route_reason = "Vague but likely rewritable query."
    elif _looks_like_greeting(query):
        route = "general"
        route_confidence = 0.95
        route_reason = "Greeting/casual message."
    elif _looks_like_math(query):
        route = "math"
        route_confidence = 0.98
        route_reason = "Math symbols/keywords detected."
    elif _looks_like_relation_query(query) and _has_any_phrase(q, rag_hints):
        route = "kg"
        route_confidence = 0.9
        route_reason = "Relation reasoning request over local/project context."
    elif _has_any_phrase(q, relation_hints) and _has_any_phrase(q, rag_hints):
        route = "kg"
        route_confidence = 0.88
        route_reason = "Relationship intent with local context hints detected."
    elif _has_any_phrase(q, web_hints) and _has_any_phrase(q, rag_hints):
        route = "hybrid"
        route_confidence = 0.9
        route_reason = "Both web and local/RAG hints detected."
    elif _has_any_phrase(q, web_hints):
        route = "web"
        route_confidence = 0.88
        route_reason = "Web/news/current intent keywords detected."
    elif _has_any_phrase(q, rag_hints):
        route = "rag"
        route_confidence = 0.88
        route_reason = "Local project/RAG keywords detected."
    else:
        low_signal = True

    # Hybrid router fallback: only for low-signal cases after deterministic rules.
    if HYBRID_ROUTER_ENABLE and low_signal and query:
        llm_route, llm_conf, llm_reason = await _llm_route_classify(query, config)
        if llm_conf >= HYBRID_ROUTER_MIN_CONFIDENCE:
            route = llm_route
            route_confidence = llm_conf
            route_reason = f"LLM classifier: {llm_reason}"
        else:
            route_reason = f"Rule fallback to general; LLM classifier confidence {llm_conf:.2f} below threshold."

    return {
        "route": route,
        "route_confidence": route_confidence,
        "route_reason": route_reason,
    }


async def clarification_agent(state: AgentState, config: RunnableConfig):
    query = (state.get("query") or "").strip()
    q = query.lower()

    if any(k in q for k in ["news", "latest", "update", "updates"]):
        prompt = (
            "I can help with that. Which topic should I focus on: "
            "AI, business, politics, sports, or local news?"
        )
    elif "help" in q:
        prompt = "Sure. What exactly do you want help with? Give one clear goal."
    else:
        prompt = (
            "Can you clarify what you mean? "
            "Please provide the exact topic and what output you want."
        )

    final = _finalize_user_output(prompt, state)
    return {
        "final_response": final,
        "messages": [AIMessage(content=final)],
        "answer_source_meta": "",
    }


async def query_rewriter_agent(state: AgentState, config: RunnableConfig):
    query = state.get("query", "").strip()

    # Prefer deterministic rewrite for known vague patterns.
    rule_based = _rule_based_rewrite(query)
    if rule_based:
        return {
            "rewritten_query": rule_based,
            "rewrite_done": True,
            "rewrite_notes": f"Rule-based rewrite: {rule_based}",
        }

    rewritten = await _call_llm(
        (
            "You rewrite vague user requests into one clear, specific query.\n"
            "Rules:\n"
            "- Keep original intent.\n"
            "- Add missing context only when obvious from user wording.\n"
            "- Return only one rewritten query sentence.\n"
        ),
        f"Original user query: {query}",
        config,
    )
    rewritten_query = (rewritten or "").strip().strip('"')
    if not rewritten_query:
        rewritten_query = query

    return {
        "rewritten_query": rewritten_query,
        "rewrite_done": True,
        "rewrite_notes": f"Rewritten query: {rewritten_query}",
    }


async def recency_guard_agent(state: AgentState, config: RunnableConfig):
    route = (state.get("route") or "").lower()
    query = _active_query(state)
    control = _parse_web_hitl_control_message(_latest_user_query(state))
    if route not in {"web", "hybrid"}:
        return {"recency_query": query, "recency_days": 0, "recency_notes": "Not required for this route."}

    pending_query = (state.get("web_hitl_pending_query") or "").strip()
    if pending_query:
        pending_recency_query = (state.get("web_hitl_pending_recency_query") or "").strip() or pending_query
        pending_days = int(state.get("web_hitl_pending_recency_days") or 0)
        return {
            "recency_query": pending_recency_query,
            "recency_days": pending_days,
            "recency_notes": "Using pending HITL web query context.",
        }

    days = _extract_recency_days(query)
    if control and str(control.get("action")) == "approve":
        control_days = int(control.get("recency_days") or 0)
        if control_days > 0:
            days = control_days
    if days is None:
        return {"recency_query": query, "recency_days": 0, "recency_notes": "No recency constraint applied."}

    guarded_query = query
    notes = f"Recency preference: prioritize last {days} days (fallback to most recent if needed)."
    return {"recency_query": guarded_query, "recency_days": days, "recency_notes": notes}


def _next_node_from_route(state: AgentState) -> str:
    route = (state.get("route") or "general").lower()
    match route:
        case "clarify":
            return "clarification_agent"
        case "rewrite":
            return "query_rewriter_agent"
        case "math":
            return "math_agent"
        case "web":
            return "recency_guard_agent"
        case "kg":
            return "knowledge_graph_agent"
        case "rag":
            return "rag_agent"
        case "hybrid":
            return "recency_guard_agent"
        case _:
            return "response_agent"


def _next_node_after_web_hitl(state: AgentState) -> str:
    decision = (state.get("web_hitl_decision") or "").lower()
    route = (state.get("route") or "").lower()

    if decision in {"awaiting", "rejected"}:
        return "evaluation_agent"
    if decision == "approved":
        web_notes = (state.get("web_notes") or "").strip()
        has_preview_notes = bool(web_notes and "not required for this route" not in web_notes.lower())
        if has_preview_notes:
            if route == "hybrid":
                return "rag_agent"
            return "response_agent"
        return "web_search_agent"

    # default path when HITL is not required for this query
    return "web_search_agent"


def _next_node_after_web(state: AgentState) -> str:
    route = (state.get("route") or "").lower()
    if route == "hybrid":
        return "rag_agent"
    return "response_agent"


def _parse_web_notes_entries(web_notes: str) -> list[dict[str, str]]:
    """
    Parse web notes produced by perform_web_search() into structured entries.

    Expected block format:
    - <title>
      Link: <url>
      Snippet: <text>
    """
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for raw_line in (web_notes or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            if current and current.get("title") and current.get("url"):
                entries.append(current)
            current = {"title": line[2:].strip(), "url": "", "date": "", "snippet": ""}
            continue
        if current is None:
            continue
        if line.lower().startswith("link:"):
            current["url"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("date:"):
            current["date"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("snippet:"):
            current["snippet"] = line.split(":", 1)[1].strip()

    if current and current.get("title") and current.get("url"):
        entries.append(current)

    return entries


async def web_search_agent(state: AgentState, config: RunnableConfig):
    query = _web_query(state)
    relevance_query = _active_query(state)
    recency_days = int(state.get("recency_days") or 0)
    route = state.get("route", "general")
    if route not in {"web", "hybrid"}:
        return {"web_notes": "Not required for this route.", "web_source_meta": ""}

    mcp_notes, _mcp_error = await mcp_web_search(
        query=query,
        max_results=5,
        recency_days=recency_days if recency_days > 0 else None,
        relevance_query=relevance_query,
    )
    if mcp_notes:
        return {"web_notes": mcp_notes, "web_source_meta": "web via mcp"}

    try:
        web_result = perform_web_search(
            query,
            max_results=5,
            recency_days=recency_days if recency_days > 0 else None,
            relevance_query=relevance_query,
            return_meta=True,
        )
        web_meta_label = ""
        if isinstance(web_result, tuple):
            web_notes, web_meta = web_result
            web_notes = (web_notes or "").strip()
            cache_hit = bool(web_meta.get("cache_hit"))
            cache_backend = str(web_meta.get("cache_backend") or "")
            if cache_hit:
                web_meta_label = f"web via {cache_backend} cache"
            else:
                web_meta_label = ""
        else:
            web_notes = str(web_result).strip()
        if not web_notes:
            web_notes = "No web results returned."
    except Exception as e:
        web_notes = f"Web retrieval failed: {e}"
        web_meta_label = ""
    return {"web_notes": web_notes, "web_source_meta": web_meta_label}


async def web_hitl_gate_agent(state: AgentState, config: RunnableConfig):
    route = (state.get("route") or "").lower()
    if route not in {"web", "hybrid"}:
        return {"web_hitl_decision": "not_required"}

    control = _parse_web_hitl_control_message(_latest_user_query(state))
    query = _active_query(state)
    recency_query = _web_query(state)
    recency_days = int(state.get("recency_days") or 0)
    pending_query = (state.get("web_hitl_pending_query") or "").strip()

    if pending_query:
        if control:
            decision = "approved" if str(control.get("action")) == "approve" else "rejected"
            reason = str(control.get("reason") or "").strip()
        else:
            user_reply = _latest_user_query(state)
            decision, reason = _parse_hitl_decision(user_reply)

        if decision == "approved":
            approved_query = pending_query
            approved_route = (state.get("web_hitl_pending_route") or route or "web").strip().lower()
            if approved_route not in {"web", "hybrid"}:
                approved_route = "web"
            approved_recency_query = (
                (state.get("web_hitl_pending_recency_query") or "").strip() or approved_query
            )
            approved_days = int(state.get("web_hitl_pending_recency_days") or 0)
            approved_notes = (state.get("web_hitl_pending_web_notes") or "").strip()
            approved_source_meta = (state.get("web_hitl_pending_source_meta") or "").strip()
            return {
                "web_hitl_decision": "approved",
                "web_hitl_reject_reason": "",
                "web_hitl_audit_query": approved_query,
                "route": approved_route,
                "query": approved_query,
                "rewritten_query": approved_query,
                "recency_query": approved_recency_query,
                "recency_days": approved_days,
                "web_notes": approved_notes,
                "web_source_meta": approved_source_meta,
                "web_hitl_pending_query": "",
                "web_hitl_pending_route": "",
                "web_hitl_pending_recency_query": "",
                "web_hitl_pending_recency_days": 0,
                "web_hitl_pending_web_notes": "",
                "web_hitl_pending_source_meta": "",
            }

        if decision == "rejected":
            reject_msg = (
                "Web search results rejected.\n\n"
                "What should I change for the next try? "
                "Tell me a tighter topic, source preference, or time window."
            )
            if reason:
                reject_msg = f"{reject_msg}\n\nReason: {reason}"
            final = _finalize_user_output(reject_msg, state)
            return {
                "web_hitl_decision": "rejected",
                "web_hitl_reject_reason": reason,
                "web_hitl_audit_query": pending_query,
                "web_hitl_pending_query": "",
                "web_hitl_pending_route": "",
                "web_hitl_pending_recency_query": "",
                "web_hitl_pending_recency_days": 0,
                "web_hitl_pending_web_notes": "",
                "web_hitl_pending_source_meta": "",
                "final_response": final,
                "messages": [AIMessage(content=final)],
            }

        reminder = _finalize_user_output(
            "A web HITL decision is pending. Reply with `approve` or `reject: <reason>`.",
            state,
        )
        return {
            "web_hitl_decision": "awaiting",
            "final_response": reminder,
            "messages": [AIMessage(content=reminder)],
        }

    # Fallback when checkpoint state is missing but UI sent explicit HITL control.
    if control and str(control.get("action")) == "approve":
        approved_query = str(control.get("query") or "").strip() or query
        approved_route = str(control.get("route") or route or "web").strip().lower()
        if approved_route not in {"web", "hybrid"}:
            approved_route = "web"
        approved_days = int(control.get("recency_days") or 0) or recency_days
        return {
            "web_hitl_decision": "approved",
            "web_hitl_reject_reason": "",
            "web_hitl_audit_query": approved_query,
            "route": approved_route,
            "query": approved_query,
            "rewritten_query": approved_query,
            "recency_query": approved_query,
            "recency_days": approved_days,
            "web_notes": "",
            "web_source_meta": "",
            "web_hitl_pending_query": "",
            "web_hitl_pending_route": "",
            "web_hitl_pending_recency_query": "",
            "web_hitl_pending_recency_days": 0,
            "web_hitl_pending_web_notes": "",
            "web_hitl_pending_source_meta": "",
        }

    if control and str(control.get("action")) == "reject":
        reason = str(control.get("reason") or "").strip()
        rejected_query = str(control.get("query") or "").strip() or query
        reject_msg = (
            "Web search results rejected.\n\n"
            "What should I change for the next try? "
            "Tell me a tighter topic, source preference, or time window."
        )
        if reason:
            reject_msg = f"{reject_msg}\n\nReason: {reason}"
        final = _finalize_user_output(reject_msg, state)
        return {
            "web_hitl_decision": "rejected",
            "web_hitl_reject_reason": reason,
            "web_hitl_audit_query": rejected_query,
            "web_hitl_pending_query": "",
            "web_hitl_pending_route": "",
            "web_hitl_pending_recency_query": "",
            "web_hitl_pending_recency_days": 0,
            "web_hitl_pending_web_notes": "",
            "web_hitl_pending_source_meta": "",
            "final_response": final,
            "messages": [AIMessage(content=final)],
        }

    # Guardrail for malformed control text.
    if "web_hitl" in (query or "").lower():
        final = _finalize_user_output(
            (
                "I received an invalid HITL control payload.\n\n"
                "Please retry your original query to regenerate preview, then click Approve/Reject again."
            ),
            state,
        )
        return {
            "web_hitl_decision": "rejected",
            "web_hitl_reject_reason": "invalid control payload",
            "web_hitl_audit_query": "",
            "final_response": final,
            "messages": [AIMessage(content=final)],
        }

    if not GRAPH_WEB_HITL_ENABLED or not _looks_like_web_hitl_candidate(query, recency_days):
        return {"web_hitl_decision": "not_required"}

    preview_source_meta = ""
    try:
        preview_result = perform_web_search(
            recency_query,
            max_results=GRAPH_WEB_HITL_MAX_RESULTS,
            recency_days=recency_days if recency_days > 0 else None,
            relevance_query=query,
            return_meta=True,
        )
        if isinstance(preview_result, tuple):
            web_notes, preview_meta = preview_result
            web_notes = (web_notes or "").strip()
            source = str(preview_meta.get("source") or "").strip()
            cache_hit = bool(preview_meta.get("cache_hit"))
            if source:
                preview_source_meta = source
            elif cache_hit:
                preview_source_meta = "web cache"
        else:
            web_notes = str(preview_result).strip()
        if not web_notes:
            web_notes = "No web preview results returned."
    except Exception as e:
        web_notes = f"Web retrieval failed: {e}"

    entries = _parse_web_notes_entries(web_notes)
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=max(1, recency_days))
        if recency_days > 0
        else None
    )
    all_within = True
    has_dated = False
    for entry in entries:
        parsed_dt = _parse_ymd_date(entry.get("date", ""))
        if parsed_dt and cutoff:
            has_dated = True
            if parsed_dt < cutoff:
                all_within = False
                break
    if not has_dated:
        all_within = False

    prompt = _format_web_hitl_preview(
        query=query,
        recency_days=recency_days,
        entries=entries,
        all_within_recency=all_within,
    )
    final = _finalize_user_output(prompt, state)
    return {
        "web_hitl_decision": "awaiting",
        "web_hitl_reject_reason": "",
        "web_hitl_pending_query": query,
        "web_hitl_pending_route": route,
        "web_hitl_pending_recency_query": recency_query,
        "web_hitl_pending_recency_days": recency_days,
        "web_hitl_pending_web_notes": web_notes,
        "web_hitl_pending_source_meta": preview_source_meta,
        "web_hitl_preview_count": len(entries),
        "web_hitl_all_within_recency": all_within,
        "final_response": final,
        "messages": [AIMessage(content=final)],
    }


async def knowledge_graph_agent(state: AgentState, config: RunnableConfig):
    query = _strip_local_prefix(_active_query(state))
    route = state.get("route", "general")
    if route != "kg":
        return {"kg_notes": "Not required for this route.", "kg_source_meta": ""}

    try:
        kg_result = query_knowledge_graph(query=query, limit=4, return_meta=True)
        kg_meta_label = "kg via local graph"
        if isinstance(kg_result, tuple):
            kg_notes, kg_meta = kg_result
            rag_meta = kg_meta.get("rag_meta", {}) if isinstance(kg_meta, dict) else {}
            cache_hit = bool(rag_meta.get("cache_hit"))
            cache_backend = str(rag_meta.get("cache_backend") or "")
            if cache_hit and cache_backend:
                kg_meta_label = f"kg via {cache_backend} cache"
        else:
            kg_notes = str(kg_result)
    except Exception as e:
        kg_notes = f"Knowledge graph retrieval failed: {e}"
        kg_meta_label = ""

    return {
        "kg_notes": kg_notes,
        "kg_source_meta": kg_meta_label,
    }


async def rag_agent(state: AgentState, config: RunnableConfig):
    query = _strip_local_prefix(_active_query(state))
    route = state.get("route", "general")
    if route not in {"rag", "hybrid", "kg"}:
        return {"rag_notes": "Not required for this route.", "rag_source_meta": ""}

    try:
        rag_result = search_local_knowledge(query, limit=3, return_meta=True)
        rag_meta_label = ""
        if isinstance(rag_result, tuple):
            rag_notes, rag_meta = rag_result
            cache_hit = bool(rag_meta.get("cache_hit"))
            cache_backend = str(rag_meta.get("cache_backend") or "")
            source = str(rag_meta.get("source") or "")
            if cache_hit:
                rag_meta_label = f"rag via {cache_backend} cache"
            else:
                rag_meta_label = ""
        else:
            rag_notes = str(rag_result)
    except Exception as e:
        rag_notes = f"Local RAG retrieval failed: {e}"
        rag_meta_label = ""
    return {"rag_notes": rag_notes, "rag_source_meta": rag_meta_label}


async def math_agent(state: AgentState, config: RunnableConfig):
    route = state.get("route", "general")
    query = _active_query(state)
    if route != "math":
        return {"math_result": "Not required for non-math route."}

    expr = _extract_expression(query)
    if not expr:
        return {"math_result": "Could not extract a valid math expression."}

    mcp_value, mcp_error = await mcp_calculator(expr)
    if mcp_value is not None and mcp_value != "":
        return {"math_result": f"{expr} = {mcp_value}", "answer_source_meta": "calculator via mcp"}

    try:
        value = numexpr.evaluate(expr, global_dict={}, local_dict={"pi": 3.141592653589793, "e": 2.718281828459045})
        math_result = f"{expr} = {str(value).strip('[]')}"
    except Exception as e:
        math_result = f"Math evaluation failed for '{expr}': {e}"
        if mcp_error:
            math_result += f" (MCP error: {mcp_error})"
    return {"math_result": math_result, "answer_source_meta": "calculator via local"}


async def response_agent(state: AgentState, config: RunnableConfig):
    safety: LlamaGuardOutput = state.get("safety")
    preserved_source_meta = (state.get("answer_source_meta") or "").strip()
    if safety and safety.safety_assessment == SafetyAssessment.UNSAFE:
        unsafe = ", ".join(safety.unsafe_categories) if safety.unsafe_categories else "unsafe content"
        final = f"I cannot help with that request because it may involve unsafe content ({unsafe})."
        final = _finalize_user_output(final, state)
        return {
            "final_response": final,
            "messages": [AIMessage(content=final)],
            "answer_source_meta": preserved_source_meta,
        }

    route = (state.get("route") or "").lower()
    web_notes = state.get("web_notes", "")
    kg_notes = state.get("kg_notes", "")
    rewritten_query = (state.get("rewritten_query") or "").strip()
    recency_notes = (state.get("recency_notes") or "").strip()
    original_query = (state.get("query") or "").strip()
    if route in {"web", "hybrid"} and web_notes.startswith("Web retrieval failed:"):
        if "no topical results matching query terms" in web_notes:
            final = (
                "I could not find recent web results that match your exact topic.\n\n"
                f"Detail: {web_notes}\n\n"
                "Try adding one or two clearer keywords (for example: "
                "'AI startup funding news this week')."
            )
            final = _finalize_user_output(final, state)
            return {
                "final_response": final,
                "messages": [AIMessage(content=final)],
                "answer_source_meta": preserved_source_meta,
            }
        if "no dated results within last" in web_notes or "no results within last" in web_notes:
            final = (
                "I could not find enough reliably dated sources inside your requested time window.\n\n"
                f"Detail: {web_notes}\n\n"
                "Try broadening the time range (for example: 'this month') or adjusting the topic keywords."
            )
            final = _finalize_user_output(final, state)
            return {
                "final_response": final,
                "messages": [AIMessage(content=final)],
                "answer_source_meta": preserved_source_meta,
            }
        final = (
            "I could not complete live web retrieval for this request.\n\n"
            f"Technical detail: {web_notes}\n\n"
            "Please retry in a moment, or ask for a non-live summary."
        )
        final = _finalize_user_output(final, state)
        return {
            "final_response": final,
            "messages": [AIMessage(content=final)],
            "answer_source_meta": preserved_source_meta,
        }

    if route in {"web", "hybrid"}:
        entries = _parse_web_notes_entries(web_notes)
        if entries:
            heading = "Here are the latest updates I found from live web sources:"
            lines = [heading]
            if recency_notes and "Not required" not in recency_notes and "No recency" not in recency_notes:
                lines.append(f"({recency_notes})")
            lines.append("")
            for i, entry in enumerate(entries[:5], start=1):
                title = entry.get("title", "").strip() or "Untitled"
                url = entry.get("url", "").strip()
                date = entry.get("date", "").strip()
                snippet = entry.get("snippet", "").strip()
                if date:
                    lines.append(f"{i}. [{title}]({url}) ({date})")
                else:
                    lines.append(f"{i}. [{title}]({url})")
                if snippet:
                    lines.append(f"   - {snippet}")
            final = _finalize_user_output("\n".join(lines), state)
            return {
                "final_response": final,
                "messages": [AIMessage(content=final)],
                "answer_source_meta": preserved_source_meta,
            }
        final = (
            "I couldn't produce a source-linked web answer for that request.\n\n"
            "Please retry, or ask with a narrower scope (for example: "
            "'latest AI startup funding news this week')."
        )
        final = _finalize_user_output(final, state)
        return {
            "final_response": final,
            "messages": [AIMessage(content=final)],
            "answer_source_meta": preserved_source_meta,
        }

    final = await _call_llm(
        base_instructions,
        (
            f"User query: {original_query}\n\n"
            f"Rewritten query: {rewritten_query}\n\n"
            f"Recency notes: {recency_notes}\n\n"
            f"Route:\n{state.get('route', '')}\n\n"
            f"Web evidence:\n{web_notes}\n\n"
            f"Knowledge graph evidence:\n{kg_notes}\n\n"
            f"Local RAG evidence:\n{state.get('rag_notes', '')}\n\n"
            f"Math result:\n{state.get('math_result', '')}\n"
        ),
        config,
    )
    final = _finalize_user_output(final, state)
    return {
        "final_response": final,
        "messages": [AIMessage(content=final)],
        "answer_source_meta": preserved_source_meta,
    }


def _count_markdown_links(text: str) -> int:
    return len(re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", text or ""))


def _evaluate_response_quality(state: AgentState) -> tuple[int, str]:
    route = (state.get("route") or "general").lower()
    final_response = (state.get("final_response") or "").strip()
    web_notes = (state.get("web_notes") or "").strip()
    rag_notes = (state.get("rag_notes") or "").strip()
    kg_notes = (state.get("kg_notes") or "").strip()
    math_result = (state.get("math_result") or "").strip()
    safety: LlamaGuardOutput | None = state.get("safety")

    score = 50
    checks: list[str] = []

    if final_response:
        score += 10
        checks.append("final_response_present:+10")
    else:
        checks.append("final_response_missing:+0")

    if safety and safety.safety_assessment == SafetyAssessment.UNSAFE:
        score -= 20
        checks.append("unsafe_content_detected:-20")
    else:
        score += 10
        checks.append("safety_ok:+10")

    if route in {"web", "hybrid"}:
        link_count = _count_markdown_links(final_response)
        if link_count >= 2:
            score += 15
            checks.append("web_citations>=2:+15")
        elif link_count == 1:
            score += 8
            checks.append("web_citations==1:+8")
        else:
            score -= 10
            checks.append("web_citations_missing:-10")

        if web_notes and not web_notes.startswith("Web retrieval failed:"):
            score += 10
            checks.append("web_retrieval_ok:+10")
        else:
            score -= 10
            checks.append("web_retrieval_failed:-10")

    if route in {"rag", "hybrid"}:
        if rag_notes and rag_notes not in {"Not required for this route."} and not rag_notes.startswith("Local RAG retrieval failed:"):
            score += 10
            checks.append("rag_retrieval_ok:+10")
        else:
            score -= 6
            checks.append("rag_retrieval_missing_or_failed:-6")

    if route == "kg":
        if kg_notes and "failed" not in kg_notes.lower() and "not required" not in kg_notes.lower():
            score += 10
            checks.append("kg_retrieval_ok:+10")
        else:
            score -= 6
            checks.append("kg_retrieval_missing_or_failed:-6")

    if route == "math":
        if math_result and "failed" not in math_result.lower():
            score += 12
            checks.append("math_result_ok:+12")
        else:
            score -= 8
            checks.append("math_result_missing_or_failed:-8")

    if route == "clarify":
        if "clarify" in final_response.lower() or "what" in final_response.lower():
            score += 8
            checks.append("clarification_prompt_quality:+8")
        else:
            checks.append("clarification_prompt_unclear:+0")

    if len(final_response) > 1800:
        score -= 5
        checks.append("response_too_long:-5")
    elif len(final_response) < 20:
        score -= 5
        checks.append("response_too_short:-5")
    else:
        score += 5
        checks.append("response_length_ok:+5")

    score = max(0, min(100, score))
    report = (
        f"Evaluation score: {score}/100 | route={route}\n"
        + "Checks: " + "; ".join(checks)
    )
    return score, report


async def evaluation_agent(state: AgentState, config: RunnableConfig):
    score, report = _evaluate_response_quality(state)
    return {"evaluation_score": score, "evaluation_report": report}


# Initialize the local knowledge store once on startup.
init_local_knowledge_store()

# Define the 12-agent orchestration graph.
agent = StateGraph(AgentState)
agent.add_node("safety_agent", safety_agent)
agent.add_node("intent_router_agent", intent_router_agent)
agent.add_node("clarification_agent", clarification_agent)
agent.add_node("query_rewriter_agent", query_rewriter_agent)
agent.add_node("recency_guard_agent", recency_guard_agent)
agent.add_node("web_hitl_gate_agent", web_hitl_gate_agent)
agent.add_node("web_search_agent", web_search_agent)
agent.add_node("knowledge_graph_agent", knowledge_graph_agent)
agent.add_node("rag_agent", rag_agent)
agent.add_node("math_agent", math_agent)
agent.add_node("response_agent", response_agent)
agent.add_node("evaluation_agent", evaluation_agent)

agent.set_entry_point("safety_agent")
agent.add_edge("safety_agent", "intent_router_agent")
agent.add_conditional_edges(
    "intent_router_agent",
    _next_node_from_route,
    {
        "recency_guard_agent": "recency_guard_agent",
        "knowledge_graph_agent": "knowledge_graph_agent",
        "rag_agent": "rag_agent",
        "math_agent": "math_agent",
        "clarification_agent": "clarification_agent",
        "query_rewriter_agent": "query_rewriter_agent",
        "response_agent": "response_agent",
    },
)
agent.add_edge("clarification_agent", "evaluation_agent")
agent.add_edge("query_rewriter_agent", "intent_router_agent")
agent.add_edge("recency_guard_agent", "web_hitl_gate_agent")
agent.add_conditional_edges(
    "web_hitl_gate_agent",
    _next_node_after_web_hitl,
    {
        "web_search_agent": "web_search_agent",
        "rag_agent": "rag_agent",
        "response_agent": "response_agent",
        "evaluation_agent": "evaluation_agent",
    },
)
agent.add_conditional_edges(
    "web_search_agent",
    _next_node_after_web,
    {
        "rag_agent": "rag_agent",
        "response_agent": "response_agent",
    },
)
agent.add_edge("knowledge_graph_agent", "rag_agent")
agent.add_edge("rag_agent", "response_agent")
agent.add_edge("math_agent", "response_agent")
agent.add_edge("response_agent", "evaluation_agent")
agent.add_edge("evaluation_agent", END)

research_assistant = agent.compile()


if __name__ == "__main__":
    import asyncio
    from uuid import uuid4
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        inputs = {"messages": [("user", "Find me the latest updates on autonomous AI agents and summarize them")]}
        result = await research_assistant.ainvoke(
            inputs,
            config=RunnableConfig(configurable={"thread_id": uuid4()}),
        )
        result["messages"][-1].pretty_print()

    asyncio.run(main())
