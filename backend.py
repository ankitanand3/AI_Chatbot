from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# --- LightRAG (simplified, without complex RAGAnything/MinerU) ---
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

load_dotenv()

# -------------------
# 0. Global config
# -------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required for RAG-Anything")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_WORKING_ROOT = os.path.join(BASE_DIR, "rag_storage")
os.makedirs(RAG_WORKING_ROOT, exist_ok=True)

# -------------------
# 1. LLM for main chat
# -------------------
llm = ChatOpenAI(model="gpt-4o-mini")

# -------------------
# 2. LightRAG setup (simplified, per-thread instances)
# -------------------
_THREAD_RAG: Dict[str, LightRAG] = {}
_THREAD_METADATA: Dict[str, dict] = {}  # thread_id -> {"filename": str}


def _build_lightrag(working_dir: str) -> LightRAG:
    """Create a simplified LightRAG instance (no complex MinerU/GPU dependencies)."""
    os.makedirs(working_dir, exist_ok=True)

    async def llm_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[list] = None,
        **kwargs: Any,
    ) -> str:
        return await openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=OPENAI_API_KEY,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model="text-embedding-3-large",
            api_key=OPENAI_API_KEY,
        ),
    )

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )

    # Initialize storages and pipeline (required)
    async def init():
        await rag.initialize_storages()
        await initialize_pipeline_status()

    asyncio.run(init())

    return rag


def _get_rag_for_thread(thread_id: str) -> LightRAG:
    """Return (or lazily create) the LightRAG instance for a given thread."""
    key = str(thread_id)
    if key not in _THREAD_RAG:
        thread_dir = os.path.join(RAG_WORKING_ROOT, key)
        _THREAD_RAG[key] = _build_lightrag(thread_dir)
    return _THREAD_RAG[key]


def _extract_text_from_document(file_bytes: bytes, filename: str) -> str:
    """Simple text extraction without complex dependencies."""
    import io
    from pypdf import PdfReader

    file_ext = os.path.splitext(filename)[1].lower()

    try:
        if file_ext == '.pdf':
            # Extract text from PDF
            reader = PdfReader(io.BytesIO(file_bytes))
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text())
            return "\n\n".join(text_parts)

        elif file_ext in ['.txt', '.md', '.csv']:
            # Direct text files
            return file_bytes.decode('utf-8', errors='ignore')

        else:
            # Try to decode as text
            return file_bytes.decode('utf-8', errors='ignore')

    except Exception as e:
        raise ValueError(f"Could not extract text from {filename}: {e}")


def ingest_document(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Ingest document text using simplified LightRAG (fast, no GPU dependencies).
    Supports: PDF, TXT, MD, CSV
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    if not filename:
        filename = f"uploaded-{thread_id}.bin"

    try:
        logger.info(f"Starting document ingestion: {filename}")

        # Extract text
        logger.info(f"Extracting text from {filename}...")
        text_content = _extract_text_from_document(file_bytes, filename)
        logger.info(f"Extracted {len(text_content)} characters")

        # Get RAG instance and insert
        rag = _get_rag_for_thread(str(thread_id))
        logger.info(f"Inserting into LightRAG knowledge base...")
        asyncio.run(rag.ainsert(text_content))
        logger.info(f"Document processing completed: {filename}")

        meta = {"filename": filename, "chars": len(text_content)}
        _THREAD_METADATA[str(thread_id)] = meta
        return meta

    except Exception as e:
        logger.error(f"Error processing document {filename}: {e}")
        import traceback
        traceback.print_exc()
        raise


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None, mode: str = "hybrid") -> dict:
    """
    Retrieve relevant information from the RAG knowledge base for the given chat thread.

    - Always include thread_id when calling this tool.
    - mode: retrieval mode (e.g. 'hybrid', 'local', 'global', 'naive').
    """
    if thread_id is None:
        return {
            "error": "rag_tool requires a thread_id",
            "query": query,
        }

    try:
        rag = _get_rag_for_thread(str(thread_id))
        result = asyncio.run(
            rag.aquery(
                query,
                param=QueryParam(mode=mode),
            )
        )
        return {
            "query": query,
            "mode": mode,
            "result": str(result),
            "thread_id": str(thread_id),
        }
    except Exception as e:
        return {
            "error": f"RAG query failed: {e}",
            "query": query,
            "thread_id": str(thread_id),
        }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer directly or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant.\n\n"
            "For questions about documents the user has uploaded in THIS chat, "
            "use the `rag_tool`. Always pass the correct `thread_id` argument so "
            "the right per-chat multimodal index is used. "
            "The RAG system is powered by RAG-Anything, which can understand "
            "text, images, tables, and equations inside the documents.\n\n"
            "You can also use the web search, stock price, and calculator tools "
            "when helpful. If no document is available for this chat, ask the "
            "user to upload one."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_METADATA


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})