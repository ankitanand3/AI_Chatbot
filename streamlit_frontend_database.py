from dotenv import load_dotenv
load_dotenv()

import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph_tool_backend import chatbot, retrieve_all_threads


# ------------------------------ utility functions -----------------------------


def generate_thread_id():
    # Keep as uuid.UUID to match existing DB records
    return uuid.uuid4()


def load_conversation(thread_id):
    """Load the message history for a thread from LangGraph.

    If LangGraph has no state for this thread (e.g. brand new or after a restart),
    just return an empty list instead of raising KeyError.
    """
    snapshot = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # snapshot.values is a dict-like; use .get to avoid KeyError
    return snapshot.values.get("messages", [])


def reset_chat():
    """Start a brand-new empty thread.

    NOTE: We do NOT register it in chat_threads yet.
    A thread only appears in 'My Conversation' after the user sends the first message.
    """
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []


def add_thread(thread_id):
    """Keep a unique ordered list of all threads that actually have messages."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# ------------------------------  Session setup ---------------------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# 1) Load existing threads from SQLite once, on first run
if "chat_threads" not in st.session_state:
    # list of thread_ids that have at least one message (from the DB)
    st.session_state["chat_threads"] = retrieve_all_threads()

# 2) Build / restore titles for those threads from their first user message
if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

    for thread_id in st.session_state["chat_threads"]:
        messages = load_conversation(thread_id)

        title = "New Chat"
        for msg in messages:
            if isinstance(msg, HumanMessage):
                clean = msg.content.replace("\n", " ").strip()
                max_len = 25
                if len(clean) > max_len:
                    clean = clean[:max_len].rstrip() + "…"
                title = clean or "New Chat"
                break  # only first user message matters

        st.session_state["chat_titles"][thread_id] = title


# -------------------------- Sidebar UI -----------------------------------

st.sidebar.title("Ankit Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversation")

# Only iterate over threads that actually have messages
for thread_id in st.session_state["chat_threads"][::-1]:
    label = st.session_state["chat_titles"].get(thread_id, "New Chat")

    # Use thread_id as the key so Streamlit can tell buttons apart
    if st.sidebar.button(label, key=str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            temp_messages.append({"role": role, "content": msg.content})

        st.session_state["message_history"] = temp_messages


# -------------------------  Main UI ----------------------------------

# Render existing messages for the active thread
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    current_thread = st.session_state["thread_id"]

    # 1) Register this thread in the list ONLY ON FIRST MESSAGE
    add_thread(current_thread)

    # 2) Set a title for this thread if it doesn’t have one yet
    if current_thread not in st.session_state["chat_titles"]:
        clean = user_input.replace("\n", " ").strip()
        max_len = 25
        if len(clean) > max_len:
            clean = clean[:max_len].rstrip() + "…"
        st.session_state["chat_titles"][current_thread] = clean or "New Chat"

    # 3) Show the user message immediately
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.text(user_input)

    # 4) Stream the assistant reply, tied to this thread_id for LangGraph
    CONFIG = {
        "configurable": {"thread_id": current_thread},
        "metadata": {"thread_id": current_thread},
        "run_name": "chat_turn",
    }

    # Assistant streaming block with tool status indicator
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
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

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )