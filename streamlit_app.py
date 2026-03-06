import asyncio
import os
import re
from typing import AsyncGenerator, List
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import get_script_run_ctx
from client import AgentClient
from schema import ChatMessage, model_dump_compat, model_validate_compat


# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Research Assistant"
APP_ICON = "🔎"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT_DIR, ".env"))
USER_AUTH_ENABLED = os.getenv("ENABLE_USER_AUTH", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
WEB_HITL_ENABLED_DEFAULT = os.getenv("WEB_HITL_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WEB_HITL_MAX_RESULTS = max(1, min(int(os.getenv("WEB_HITL_MAX_RESULTS", "5")), 10))


def _history_rows_to_chat_messages(rows: list[dict]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for row in reversed(rows or []):
        role = str(row.get("role") or "").strip()
        if role not in {"human", "ai", "tool"}:
            continue
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        messages.append(
            ChatMessage(
                type=role,  # type: ignore[arg-type]
                content=content,
                run_id=str(row.get("run_id") or "") or None,
            )
        )
    return messages


def _thread_label(thread: dict) -> str:
    thread_id = str(thread.get("thread_id") or "")
    count = int(thread.get("message_count") or 0)
    when = str(thread.get("last_message_at") or "")
    preview = str(thread.get("last_message_preview") or "").replace("\n", " ").strip()
    if len(preview) > 64:
        preview = preview[:61] + "..."
    when_short = when.replace("T", " ")[:19] if when else "-"
    if preview:
        return f"{when_short} | {count} msgs | {preview}"
    return f"{when_short} | {count} msgs | {thread_id}"


def _extract_hitl_recency_days(query: str) -> int:
    q = (query or "").strip().lower()
    if not q:
        return 0
    if "today" in q or "past 24 hour" in q or "last 24 hour" in q:
        return 1

    direct_match = re.search(r"\b(?:last|past)\s+(\d{1,3})\s+day", q)
    if direct_match:
        return max(1, min(int(direct_match.group(1)), 30))

    week_match = re.search(r"\b(?:last|past)\s+(\d{1,2})\s+week", q)
    if week_match:
        return max(1, min(int(week_match.group(1)) * 7, 30))

    if "last week" in q or "past week" in q or "this week" in q:
        return 7
    if "last month" in q or "past month" in q or "this month" in q:
        return 30

    if any(token in q for token in ("latest", "recent", "current", "news", "updates")):
        return 7
    return 0


def _is_web_hitl_candidate(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    if q.startswith("local:"):
        return False
    if any(token in q for token in ("calculate", "equation", "math")):
        return False
    if re.fullmatch(r"[\d\.\s\+\-\*/\(\)\^=]+", q):
        return False

    recency_terms = (
        "latest",
        "recent",
        "current",
        "today",
        "this week",
        "last week",
        "past",
    )
    web_terms = ("news", "update", "updates", "headline", "headlines", "happening", "trend", "trends")
    return any(term in q for term in recency_terms) and any(term in q for term in web_terms)

@st.cache_resource
def get_agent_client():
    agent_url = os.getenv("AGENT_URL") or os.getenv("API_BASE_URL", "http://localhost:8000")
    return AgentClient(agent_url)


def get_available_models() -> dict[str, str]:
    models: dict[str, str] = {}
    if os.getenv("OPENAI_API_KEY"):
        models["OpenAI GPT-4o-mini (streaming)"] = "gpt-4o-mini"
    if os.getenv("GROQ_API_KEY"):
        models["llama-3.1-70b on Groq"] = "llama-3.1-70b"
    return models


async def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    models = get_available_models()
    if not models:
        st.error(
            "No model API key found. Add OPENAI_API_KEY or GROQ_API_KEY to .env and restart Streamlit."
        )
        st.code("OPENAI_API_KEY=your_key_here\nGROQ_API_KEY=your_key_here")
        st.stop()

    agent_client = get_agent_client()
    if USER_AUTH_ENABLED:
        if "auth_user_id" not in st.session_state:
            st.session_state.auth_user_id = ""
        if "auth_token" not in st.session_state:
            st.session_state.auth_token = ""
        if "thread_summaries" not in st.session_state:
            st.session_state.thread_summaries = []
        if st.session_state.auth_token:
            agent_client.set_access_token(st.session_state.auth_token)

    if "web_hitl_enabled" not in st.session_state:
        st.session_state.web_hitl_enabled = WEB_HITL_ENABLED_DEFAULT
    if "pending_web_hitl" not in st.session_state:
        st.session_state.pending_web_hitl = None
    if "web_hitl_reject_reason" not in st.session_state:
        st.session_state.web_hitl_reject_reason = ""

    ctx = get_script_run_ctx()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = (getattr(ctx, "session_id", None) if ctx else None) or str(uuid4())
    thread_id = st.session_state.thread_id

    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")
        if USER_AUTH_ENABLED:
            st.subheader("Account")
            if st.session_state.auth_user_id:
                st.success(f"Signed in: `{st.session_state.auth_user_id}`")
                col_new, col_refresh = st.columns(2)
                if col_new.button("New Chat"):
                    st.session_state.messages = []
                    st.session_state.thread_id = str(uuid4())
                    st.session_state.pending_web_hitl = None
                    st.session_state.web_hitl_reject_reason = ""
                    st.rerun()
                if col_refresh.button("Refresh"):
                    try:
                        threads_payload = await agent_client.alist_threads(limit=30)
                        st.session_state.thread_summaries = threads_payload.get("threads", [])
                        st.toast("Conversation list refreshed.")
                        st.rerun()
                    except Exception as e:
                        st.warning(f"Could not refresh history: {e}")

                thread_summaries = st.session_state.get("thread_summaries", [])
                thread_ids = [str(t.get("thread_id") or "") for t in thread_summaries if t.get("thread_id")]
                if thread_ids:
                    if st.session_state.thread_id not in thread_ids:
                        st.session_state.thread_id = thread_ids[0]
                    label_map = {
                        str(t.get("thread_id") or ""): _thread_label(t) for t in thread_summaries
                    }
                    selected_thread = st.selectbox(
                        "Conversation History",
                        options=thread_ids,
                        index=thread_ids.index(st.session_state.thread_id),
                        format_func=lambda tid: label_map.get(tid, tid),
                    )
                    if selected_thread != st.session_state.thread_id:
                        try:
                            history_payload = await agent_client.aget_store(selected_thread, limit=200)
                            st.session_state.thread_id = selected_thread
                            st.session_state.messages = _history_rows_to_chat_messages(
                                history_payload.get("messages", [])
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load selected conversation: {e}")

                if st.button("Logout"):
                    st.session_state.auth_user_id = ""
                    st.session_state.auth_token = ""
                    st.session_state.messages = []
                    st.session_state.thread_summaries = []
                    st.session_state.pending_web_hitl = None
                    st.session_state.web_hitl_reject_reason = ""
                    if "thread_id" in st.session_state:
                        del st.session_state.thread_id
                    agent_client.set_access_token(None)
                    st.rerun()
            else:
                auth_mode = st.radio(
                    "Authentication",
                    options=["Login", "Register"],
                    horizontal=True,
                )
                auth_user_id = st.text_input("User ID", key="auth_user_id_input")
                auth_password = st.text_input("Password", type="password", key="auth_password_input")
                if st.button("Continue"):
                    try:
                        if auth_mode == "Register":
                            auth = await agent_client.aregister(auth_user_id, auth_password)
                        else:
                            auth = await agent_client.alogin(auth_user_id, auth_password)
                        st.session_state.auth_user_id = auth.user_id
                        st.session_state.auth_token = auth.access_token
                        agent_client.set_access_token(auth.access_token)
                        st.session_state.pending_web_hitl = None
                        st.session_state.web_hitl_reject_reason = ""

                        threads_payload = await agent_client.alist_threads(limit=30)
                        thread_summaries = threads_payload.get("threads", [])
                        st.session_state.thread_summaries = thread_summaries
                        if thread_summaries:
                            latest_thread_id = str(thread_summaries[0].get("thread_id") or "")
                            if latest_thread_id:
                                history_payload = await agent_client.aget_store(latest_thread_id, limit=200)
                                st.session_state.thread_id = latest_thread_id
                                st.session_state.messages = _history_rows_to_chat_messages(
                                    history_payload.get("messages", [])
                                )
                                st.success("Authentication successful. Loaded latest conversation history.")
                            else:
                                st.session_state.messages = []
                                st.session_state.thread_id = str(uuid4())
                                st.success("Authentication successful. Started a new conversation.")
                        else:
                            st.session_state.messages = []
                            st.session_state.thread_id = str(uuid4())
                            st.success("Authentication successful. Started a new conversation.")
                        await asyncio.sleep(0.1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Authentication failed: {e}")

            if not st.session_state.auth_user_id:
                st.info("Login or register to start chatting.")

        st.caption("Thread ID (for /store/{thread_id})")
        st.code(thread_id)
        with st.popover(":material/settings: Settings"):
            m = st.radio("LLM to use", options=list(models.keys()))
            model = models[m]
            use_streaming = st.toggle("Stream results", value=True)
            st.session_state.web_hitl_enabled = st.toggle(
                "HITL for latest web news",
                value=bool(st.session_state.web_hitl_enabled),
                help="For recency/news queries, show web results first and require approve/reject before final answer.",
            )
        with st.popover(":material/policy: Privacy"):
            st.write("Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only.")

    if USER_AUTH_ENABLED and not st.session_state.auth_user_id:
        st.stop()

    thread_id = st.session_state.thread_id

    # Draw existing messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    messages: List[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        WELCOME = "Hello! I'm an AI assistant with web search, local RAG retrieval, and a calculator. Ask me anything!"
        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter():
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    async def _submit_user_query(input_text: str) -> None:
        messages.append(ChatMessage(type="human", content=input_text))
        st.chat_message("human").write(input_text)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=input_text,
                    model=model,
                    thread_id=thread_id,
                )
                await draw_messages(stream, is_new=True)
            else:
                response = await agent_client.ainvoke(
                    message=input_text,
                    model=model,
                    thread_id=thread_id,
                )
                messages.append(response)
                st.chat_message("ai").write(response.content)

            # Keep local thread summary fresh after successful send.
            if USER_AUTH_ENABLED and st.session_state.get("auth_user_id"):
                try:
                    threads_payload = await agent_client.alist_threads(limit=30)
                    st.session_state.thread_summaries = threads_payload.get("threads", [])
                except Exception:
                    pass
        except Exception as e:
            err = f"Request failed: {e}"
            st.chat_message("ai").error(err)
            messages.append(ChatMessage(type="ai", content=err))

    pending_hitl = st.session_state.get("pending_web_hitl")
    if pending_hitl:
        with st.chat_message("ai"):
            st.markdown("### Human Approval Needed: Web Search Results")
            st.caption(
                f"Query: {pending_hitl.get('query', '')} | Recency target: last {pending_hitl.get('recency_days', 7)} days"
            )
            if pending_hitl.get("all_within_recency"):
                st.success("All dated preview results are within your recency window.")
            else:
                st.warning("Some results may be outside the requested recency window. Review before continuing.")

            preview_items = pending_hitl.get("items", [])
            for idx, item in enumerate(preview_items, start=1):
                title = str(item.get("title") or "Untitled result")
                url = str(item.get("url") or "")
                date_text = str(item.get("published_date") or "date unknown")
                snippet = str(item.get("snippet") or "")
                recency_flag = item.get("is_within_recency")
                if recency_flag is True:
                    marker = "within range"
                elif recency_flag is False:
                    marker = "outside range"
                else:
                    marker = "date unknown"
                st.markdown(f"{idx}. **{title}**")
                st.caption(f"{date_text} | {marker}")
                if url:
                    st.markdown(f"[Open source]({url})")
                if snippet:
                    st.write(snippet)

            st.text_input(
                "Reject reason (optional)",
                key="web_hitl_reject_reason",
                placeholder="Example: Results are older than last week.",
            )
            col_approve, col_reject = st.columns(2)
            approve_clicked = col_approve.button("Approve and Continue", key="web_hitl_approve")
            reject_clicked = col_reject.button("Reject", key="web_hitl_reject")

        if approve_clicked:
            approved_query = str(pending_hitl.get("query") or "").strip()
            try:
                if approved_query:
                    await agent_client.arecord_web_hitl_decision(
                        query=approved_query,
                        decision="approved",
                        thread_id=str(st.session_state.get("thread_id") or ""),
                        recency_days=int(pending_hitl.get("recency_days") or 0),
                        preview_count=int(pending_hitl.get("count") or len(pending_hitl.get("items", []))),
                        all_within_recency=bool(pending_hitl.get("all_within_recency")),
                        source=str(pending_hitl.get("source") or ""),
                        cache_hit=bool(pending_hitl.get("cache_hit")),
                    )
            except Exception as e:
                st.warning(f"Could not persist HITL approval audit record: {e}")
            st.session_state.pending_web_hitl = None
            if approved_query:
                await _submit_user_query(approved_query)
            st.rerun()

        if reject_clicked:
            rejected_query = str(pending_hitl.get("query") or "").strip()
            reason = str(st.session_state.get("web_hitl_reject_reason") or "").strip()
            try:
                if rejected_query:
                    await agent_client.arecord_web_hitl_decision(
                        query=rejected_query,
                        decision="rejected",
                        thread_id=str(st.session_state.get("thread_id") or ""),
                        reason=reason,
                        recency_days=int(pending_hitl.get("recency_days") or 0),
                        preview_count=int(pending_hitl.get("count") or len(pending_hitl.get("items", []))),
                        all_within_recency=bool(pending_hitl.get("all_within_recency")),
                        source=str(pending_hitl.get("source") or ""),
                        cache_hit=bool(pending_hitl.get("cache_hit")),
                    )
            except Exception as e:
                st.warning(f"Could not persist HITL rejection audit record: {e}")
            st.session_state.pending_web_hitl = None
            rejection_note = "Web search results rejected. Please refine your news request."
            if reason:
                rejection_note = f"{rejection_note}\n\nReason: {reason}"
            if rejected_query:
                messages.append(ChatMessage(type="human", content=rejected_query))
            messages.append(ChatMessage(type="ai", content=rejection_note))
            st.rerun()

    input_text = st.chat_input(disabled=bool(st.session_state.get("pending_web_hitl")))
    if input_text:
        if bool(st.session_state.get("web_hitl_enabled")) and _is_web_hitl_candidate(input_text):
            recency_days = _extract_hitl_recency_days(input_text)
            if recency_days > 0:
                try:
                    preview = await agent_client.aweb_search_preview(
                        query=input_text,
                        recency_days=recency_days,
                        max_results=WEB_HITL_MAX_RESULTS,
                    )
                    if preview.get("count", 0) > 0:
                        preview["query"] = input_text
                        st.session_state.web_hitl_reject_reason = ""
                        st.session_state.pending_web_hitl = preview
                        st.rerun()
                    else:
                        messages.append(ChatMessage(
                            type="ai",
                            content=(
                                "HITL preview found no web results, so the query was not sent. "
                                "Please refine the request and try again."
                            ),
                        ))
                        st.rerun()
                except Exception as e:
                    messages.append(ChatMessage(
                        type="ai",
                        content=f"HITL preview failed, so the query was not sent.\n\nError: {e}",
                    ))
                    st.rerun()

        await _submit_user_query(input_text)
        st.rerun()  # Clear stale containers

    # If messages have been generated, show feedback widget
    if len(messages) > 0 and st.session_state.last_message is not None and messages[-1].type == "ai":
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
        messages_agen: AsyncGenerator[ChatMessage | str, None],
        is_new=False,
    ):
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.
    
    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()
            
            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            # Streamlit reloads and mixed module import paths can produce a
            # ChatMessage-like object from a different class identity.
            # Normalize to the local ChatMessage model instead of failing.
            try:
                if isinstance(msg, dict):
                    msg = model_validate_compat(ChatMessage, msg)
                else:
                    msg = model_validate_compat(ChatMessage, model_dump_compat(msg))
            except Exception:
                st.error(f"Unexpected message type: {type(msg)}")
                st.write(msg)
                st.stop()
        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)
                
                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                
                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # Create a status container for each tool call and store the
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                    f"""Tool Call: {tool_call["name"]}""",
                                    state="running" if is_new else "complete",
                                )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.write(tool_call["args"])

                        # Expect one ToolMessage for each tool call.
                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)
                            if not tool_result.type == "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()
                            
                            # Record the message if it's new, and update the correct
                            # status container with the result
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            # In case of an unexpected message type, log an error and stop
            case _: 
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback():
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)
    
    latest_run_id = st.session_state.messages[-1].run_id
    if not latest_run_id:
        return
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback and (latest_run_id, feedback) != st.session_state.last_feedback:
        
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client = get_agent_client()
        if st.session_state.get("auth_token"):
            agent_client.set_access_token(st.session_state.auth_token)
        await agent_client.acreate_feedback(
            run_id=latest_run_id,
            key="human-feedback-stars",
            score=normalized_score,
            kwargs=dict(
                comment="In-line human feedback",
            ),
        )
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


if __name__ == "__main__":
    asyncio.run(main())
