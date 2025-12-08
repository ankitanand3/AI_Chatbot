from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite
import requests
import asyncio
import threading

load_dotenv()

# ------------------- Async loop utilities -------------------
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI()

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


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


client = MultiServerMCPClient(
    {
        "expense": {
            "transport": "streamable_http",  # if this fails, try "sse"
            "url": "https://flat-beige-crayfish.fastmcp.app/mcp",
        }
    }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []


mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 3. State
# -------------------


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 4. Nodes
# -------------------


async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------


async def _alist_threads():
    """Return thread_ids in stable chronological order (oldest → newest)."""
    # Track the earliest (minimum) checkpoint_id for each thread
    thread_min_checkpoint = {}

    async for checkpoint in checkpointer.alist(None):
        cfg = checkpoint.config.get("configurable", {})
        tid = cfg.get("thread_id")
        if tid is None:
            continue

        # Get checkpoint_id from metadata
        checkpoint_id = checkpoint.metadata.get("checkpoint_id", "")

        # Keep the minimum (earliest) checkpoint_id for each thread
        if tid not in thread_min_checkpoint:
            thread_min_checkpoint[tid] = checkpoint_id
        else:
            # Compare and keep the smaller (earlier) checkpoint_id
            if checkpoint_id < thread_min_checkpoint[tid]:
                thread_min_checkpoint[tid] = checkpoint_id

    # Sort threads by their earliest checkpoint ID in descending order (newest → oldest)
    sorted_threads = sorted(
        thread_min_checkpoint.keys(),
        key=lambda t: thread_min_checkpoint[t],
        reverse=True
    )
    return sorted_threads


def retrieve_all_threads():
    # Returns newest → oldest; ready for direct display
    return run_async(_alist_threads())