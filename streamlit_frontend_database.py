import queue
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph_mcp_backend import (
    chatbot,
    retrieve_all_threads,
    submit_async_task,
)

# =========================== Utilities ===========================


def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    """Create a fresh empty thread (not shown in sidebar until first message)."""
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []


def add_thread(thread_id):
    """Track threads that actually have messages (chronological order)."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    """Load conversation from LangGraph; return [] if no messages yet."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    # oldest → newest from DB
    st.session_state["chat_threads"] = retrieve_all_threads()

if "chat_titles" not in st.session_state:
    # thread_id -> short title from first user message
    st.session_state["chat_titles"] = {}

    # rebuild titles from stored conversations
    for thread_id in st.session_state["chat_threads"]:
        messages = load_conversation(thread_id)
        title = "New Chat"
        for msg in messages:
            if isinstance(msg, HumanMessage):
                clean = msg.content.replace("\n", " ").strip()
                max_len = 25
                if len(clean) > max_len:
                    clean = clean[: max_len].rstrip() + "…"
                title = clean or "New Chat"
                break
        st.session_state["chat_titles"][thread_id] = title

# ============================ Sidebar ============================

st.sidebar.title("Ankit Chatbot")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()

st.sidebar.header("My Conversations")

# Show newest first in UI (backend already returns newest → oldest)
for thread_id in st.session_state["chat_threads"]:
    label = st.session_state["chat_titles"].get(thread_id, "New Chat")

    if st.sidebar.button(label, key=str(thread_id), use_container_width=True):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_messages.append({"role": "user", "content": msg.content})

            elif isinstance(msg, AIMessage):
                # Skip AI tool-call messages (they have tool_calls)
                if getattr(msg, "tool_calls", None):
                    continue
                temp_messages.append({"role": "assistant", "content": msg.content})

            elif isinstance(msg, ToolMessage):
                # Don't show raw tool JSON in the chat UI
                continue

        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    current_thread = st.session_state["thread_id"]

    # 1) Register thread only when there is at least one user message
    add_thread(current_thread)

    # 2) Set title if this is the first message in this thread
    if current_thread not in st.session_state["chat_titles"]:
        clean = user_input.replace("\n", " ").strip()
        max_len = 25
        if len(clean) > max_len:
            clean = clean[:max_len].rstrip() + "…"
        st.session_state["chat_titles"][current_thread] = clean or "New Chat"

    # 3) Show user message
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": current_thread},
        "metadata": {"thread_id": current_thread},
        "run_name": "chat_turn",
    }

    # 4) Assistant streaming block (async + tools)
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break

                message_chunk, metadata = item
                if message_chunk == "error":
                    raise metadata

                # Tool status indicator
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # 5) Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )