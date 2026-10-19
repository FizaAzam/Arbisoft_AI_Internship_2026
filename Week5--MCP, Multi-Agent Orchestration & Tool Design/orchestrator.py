# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain>=1.3.14",
#     "langgraph>=1.2.10",
#     "langchain-openai>=1.4.1",
#     "langchain-mcp-adapters>=0.3.1",
#     "mcp<2.0.0",
#     "python-dotenv>=1.2.2",
# ]
# ///


#Run with:  uv run orchestrator.py

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# ---------------------------------------------------------------- config

load_dotenv()  

ROOT = Path(__file__).parent
SERVER_PATH = ROOT / "server.py"

SERVER_PYTHON = str(ROOT / ".venv/Scripts/python.exe")

#willcreate client
MCP_SERVERS = {
    "task-manager": {
        "command": SERVER_PYTHON,
        "args": [str(SERVER_PATH)],
        "transport": "stdio",
    }
}


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SUPERVISOR_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
WORKER_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"


def make_llm(model: str) -> ChatOpenAI:
    """A chat model served by OpenRouter (an OpenAI-compatible endpoint)."""
    return ChatOpenAI(
        model=model,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,  #prefermostlikely
    )


# ------------------------------------------------------------ mcp bridge


SESSION = None


# langchain-mcp-adapters converts MCP tools automatically but ignores MCP resources, so tasks://all is wrapped into a tool by hand here.
@tool
async def read_tasks() -> str:
    """Read the complete to-do list.

    Returns a JSON array where each task has an id, title, priority
    (low/medium/high) and a done flag. Call this before answering any
    question about tasks, then do the filtering or counting yourself.
    """
    result = await SESSION.read_resource("tasks://all")
    return result.contents[0].text


# --------------------------------------------------------------- workers

READER_PROMPT = """You answer questions about a to-do list. You are read-only.

Always call read_tasks first - never answer from memory, because the list may
have changed. It returns a JSON array, and the filtering, counting and
summarising are your job; there is no tool for them.

Answer only what was asked, in one or two short sentences. If asked to change
anything, say you are read-only and that the writer handles changes."""

WRITER_PROMPT = """You create tasks on a to-do list. You are write-only.

Call add_task once per task. Strip filler from the title: store "Buy milk", not
"add a task to buy milk". Set priority only if the request implies one - urgent,
ASAP or important mean "high"; whenever or sometime mean "low"; otherwise leave
it out and let the default apply.

Report what you created, including the id the tool returned, in one line per
task. You cannot read the list, so never claim to know what is already on it."""


def build_workers(mcp_tools):
    """The two sub-agents. They differ by which tools they are handed."""
    add_task = next(t for t in mcp_tools if t.name == "add_task")

    reader = create_agent(
        model=make_llm(WORKER_MODEL),
        tools=[read_tasks],  
        system_prompt=READER_PROMPT,
    )
    writer = create_agent(
        model=make_llm(WORKER_MODEL),
        tools=[add_task], 
        system_prompt=WRITER_PROMPT,
    )
    return reader, writer


# ------------------------------------------------------------ tracing

TRACE_PATH = ROOT / "trace.jsonl"

CURRENT_AGENT = "unknown"


def trace_event(event: str, **fields) -> None:
    """Append one line of JSON to the trace file."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
        **fields,
    }
    with open(TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class ToolTracer(BaseCallbackHandler):

    def __init__(self) -> None:
        self.started: dict = {}

    @staticmethod
    def _agent(tags) -> str:
  
        for tag in tags or []:
            if tag in ("reader", "writer"):
                return tag
        return CURRENT_AGENT

    def on_tool_start(self, serialized, input_str, *, run_id, tags=None, **kwargs):
        self.started[run_id] = time.perf_counter()
        trace_event(
            "tool_start",
            agent=self._agent(tags),
            tool=(serialized or {}).get("name", "?"),
            args=str(input_str)[:200],
        )

    def on_tool_end(self, output, *, run_id, tags=None, **kwargs):
        began = self.started.pop(run_id, None)
        trace_event(
            "tool_end",
            agent=self._agent(tags),
            duration_ms=round((time.perf_counter() - began) * 1000) if began else None,
            result=str(output)[:200],
        )

    def on_tool_error(self, error, *, run_id, tags=None, **kwargs):
        self.started.pop(run_id, None)
        trace_event("tool_error", agent=self._agent(tags), error=str(error)[:200])


TRACER = ToolTracer()


async def run_agent(agent, request: str, name: str) -> str:

    global CURRENT_AGENT
    CURRENT_AGENT = name

    trace_event("agent_start", agent=name, request=request)
    started = time.perf_counter()

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        config={"callbacks": [TRACER], "tags": [name]},
    )
    reply = result["messages"][-1].content

    trace_event(
        "agent_end",
        agent=name,
        duration_ms=round((time.perf_counter() - started) * 1000),
        reply=reply[:200],
    )
    return reply


# ------------------------------------------------- supervisor and the graph

READER = None
WRITER = None

SUPERVISOR_PROMPT = """You route a to-do list request to the right worker.

  reader - the request ONLY asks about the list: what is pending, how many,
           which are high priority, is X on the list, summarise my list.
  writer - the request ONLY changes the list: add, create, record, note down,
           remind me to.
  both   - the request does BOTH in the same sentence.

Pick "both" whenever the request contains a change AND a question. This is a
single decision and there is no later turn, so never route to one worker
expecting the rest to be handled afterwards. "both" runs the writer first and
then the reader, so the reader sees the task that was just added.

Then split the request. Each worker sees ONLY the text you give it, so write
each part as a complete, self-contained instruction:

  "add a task to call mom, then tell me what's still pending"
     worker      -> both
     writer_task -> "add a task to call mom"
     reader_task -> "tell me what is still pending"

Leave a task empty when that worker is not involved.
Keep the reason to one short sentence."""

#route defines the structure the supervisor's answer should follow
class Route(BaseModel):
    """Which worker handles a request, and the part of the request each one gets."""

    worker: Literal["reader", "writer", "both"] = Field(description="who handles this")
    reason: str = Field(description="a few words on why")
    writer_task: str = Field(default="", description="the writer's part, or empty")
    reader_task: str = Field(default="", description="the reader's part, or empty")

#state defines the info that may travel b/w graph nodes
class State(TypedDict):

    request: str
    decision: str
    reason: str
    writer_task: str
    reader_task: str
    writer_result: str
    reader_result: str


async def supervisor_node(state: State) -> dict:

    llm = make_llm(SUPERVISOR_MODEL).with_structured_output(Route)
    route = await llm.ainvoke(
        [
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": state["request"]},
        ]
    )
    print(f"ROUTING: {route.worker} - {route.reason}")
    trace_event("routing", agent="supervisor", decision=route.worker,
                reason=route.reason, request=state["request"])

    #giving worker the originl user request if supervisor left it empty
    return {
        "decision": route.worker,
        "reason": route.reason,
        "writer_task": route.writer_task or state["request"],
        "reader_task": route.reader_task or state["request"],
    }


async def writer_node(state: State) -> dict:
    return {"writer_result": await run_agent(WRITER, state["writer_task"], "writer")}


async def reader_node(state: State) -> dict:
    return {"reader_result": await run_agent(READER, state["reader_task"], "reader")}


def build_graph():
    #On "both" the writer runs first, then the reader, so the reader sees the task that was just added.

    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("reader", reader_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda s: "writer" if s["decision"] in ("writer", "both") else "reader",
    )

    graph.add_conditional_edges(
        "writer",
        lambda s: "reader" if s["decision"] == "both" else END,
    )

    graph.add_edge("reader", END)

    return graph.compile()


# ------------------------------------------------------------ test


DEMO_REQUESTS = [
    "what's pending on my list?",
    "add a task to buy milk, high priority",
    "add a task to call mom, then tell me what's still pending",
]


async def main() -> None:
    global SESSION, READER, WRITER

    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing - check your .env file."
        )

    TRACE_PATH.unlink(missing_ok=True)  

    client = MultiServerMCPClient(MCP_SERVERS)


    async with client.session("task-manager") as session:
        SESSION = session
        mcp_tools = await load_mcp_tools(session)

        READER, WRITER = build_workers(mcp_tools)
        app = build_graph()

        for request in DEMO_REQUESTS:
            print("=" * 70)
            print("USER:", request)
            print("-" * 70)

            final = await app.ainvoke({"request": request})

            if final.get("writer_result"):
                print("  writer:", final["writer_result"])
            if final.get("reader_result"):
                print("  reader:", final["reader_result"])
            print()

        print_trace_summary()


def print_trace_summary() -> None:
    lines = TRACE_PATH.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    tool_calls = [r for r in records if r["event"] == "tool_start"]

    print("=" * 70)
    print(f"TRACE  ->  {TRACE_PATH.name}")
    print("=" * 70)
    print(f"{len(records)} events logged, {len(tool_calls)} tool calls\n")

    print(f"  {'AGENT':<11} {'TOOL':<12} {'MS':>6}  ARGS")
    ends = [r for r in records if r["event"] == "tool_end"]
    for call, end in zip(tool_calls, ends):
        print(
            f"  {call['agent']:<11} {call['tool']:<12} "
            f"{end.get('duration_ms', 0):>6}  {call['args'][:36]}"
        )


if __name__ == "__main__":
    asyncio.run(main())
