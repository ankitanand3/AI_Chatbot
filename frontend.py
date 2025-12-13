import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend import (
    chatbot,
    ingest_document,
    retrieve_all_threads,
    thread_document_metadata,
)

# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    """
    Start a brand new (empty) chat.

    IMPORTANT: We DO NOT register this thread in chat_threads yet.
    A thread only becomes a "past conversation" after the user sends
    at least one message.
    """
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    # Only threads that already have checkpoints (i.e. have seen messages)
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
                    clean = clean[:max_len].rstrip() + "…"
                title = clean or "New Chat"
                break
        st.session_state["chat_titles"][thread_id] = title

if "ingested_docs" not in st.session_state:
    # thread_id (str) -> {filename: meta}
    st.session_state["ingested_docs"] = {}

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]  # newest first
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("Ankit Chatbot")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(f"Using `{latest_doc.get('filename')}`")
else:
    st.sidebar.info("No documents indexed yet.")

uploaded_doc = st.sidebar.file_uploader(
    "Upload a document for this chat",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tiff",
        "gif",
        "webp",
        "txt",
        "md",
        "doc",
        "docx",
        "ppt",
        "pptx",
        "xls",
        "xlsx",
    ],
)
if uploaded_doc:
    if uploaded_doc.name in thread_docs:
        st.sidebar.info(f"`{uploaded_doc.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing document…", expanded=True) as status_box:
            summary = ingest_document(
                uploaded_doc.getvalue(),
                thread_id=thread_key,
                filename=uploaded_doc.name,
            )
            thread_docs[uploaded_doc.name] = summary
            status_box.update(label="✅ Document indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        label = st.session_state["chat_titles"].get(thread_id, "New Chat")
        if st.sidebar.button(label, key=f"side-thread-{thread_id}", use_container_width=True):
            selected_thread = thread_id

# ============================ Main Layout ========================
st.title("Ankit Chatbot")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    # Register thread only when there is at least one user message
    add_thread(st.session_state["thread_id"])

    # Set title if this is the first message in this thread
    if st.session_state["thread_id"] not in st.session_state["chat_titles"]:
        clean = user_input.replace("\n", " ").strip()
        max_len = 25
        if len(clean) > max_len:
            clean = clean[:max_len].rstrip() + "…"
        st.session_state["chat_titles"][st.session_state["thread_id"]] = clean or "New Chat"

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
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

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(f"Document indexed: {doc_meta.get('filename')}")

st.divider()

if selected_thread:
    # Switch current thread & rebuild visible chat history
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

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
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()