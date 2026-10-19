#Run with:  .venv\\Scripts\\python.exe orchestrator.py
import os
import sys
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from restaurant import MENU_TOOLS, ORDER_TOOLS

# ---------------------------------------------------------------- config

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SUPERVISOR_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
WORKER_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"


def make_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,  
    )


# --------------------------------------------------------------- workers

MENU_PROMPT = """You look after the restaurant's menu. You never take orders.

Call view_menu before answering any question about what is served or what
something costs - never answer from memory, because the menu may have changed.
Use add_menu_item to put a new dish on the menu; it needs a name and a price in
rupees.

The tool result is the truth. If add_menu_item says the dish is already on the
menu, tell the customer that plainly instead of claiming you added it.

Answer in one or two short sentences. If asked to place an order, say that the
ordering staff handle that."""

ORDER_PROMPT = """You take customers' orders. You never change the menu.

Call place_order once for the whole order, passing every dish in the items list,
for example [{"item_name": "Biryani", "quantity": 2}]. Use view_orders to report
what has already been ordered, and view_menu if you need to check a dish name.

place_order needs the customer's name. If you have not been told it, ask for it
rather than inventing one.

The tool result is the truth. If it reports an error - a dish that is not on the
menu, or a bad quantity - pass that back to the customer and let them correct it.
Never say an order was placed unless the tool confirmed it.

When you report existing orders, copy the lines view_orders gave you exactly as
they are. Do not reword, merge or tidy them - you will get the items wrong.

Answer in one or two short sentences, except when listing orders as above. If
asked to add a dish to the menu, say the kitchen staff handle that."""


def build_workers():
    menu_agent = create_agent(
        model=make_llm(WORKER_MODEL),
        tools=MENU_TOOLS,
        system_prompt=MENU_PROMPT,
    )
    order_agent = create_agent(
        model=make_llm(WORKER_MODEL),
        tools=ORDER_TOOLS,
        system_prompt=ORDER_PROMPT,
    )
    return menu_agent, order_agent


def run_agent(agent, request: str, name: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].content


# ------------------------------------------------- supervisor and the graph

MENU_AGENT = None
ORDER_AGENT = None

SUPERVISOR_PROMPT = """You route a restaurant request to the right member of staff.

  menu  - the request is ONLY about the menu itself: what do you serve, how much
          is X, add a new dish, what is on the menu.
  order - the request is ONLY about orders: place an order, I want two biryanis,
          what has been ordered, show me the orders.
  both  - the request does BOTH in the same sentence.

Pick "both" whenever the request touches the menu AND an order. This is a single
decision and there is no later turn, so never route to one worker expecting the
rest to be handled afterwards. "both" runs the menu worker first and then the
order worker, so a dish that was just added can be ordered straight away.

Then split the request. Each worker sees ONLY the text you give it, so write each
part as a complete, self-contained instruction:

  "add mango lassi for 200, then order two of them for Ali"
     worker     -> both
     menu_task  -> "add mango lassi to the menu for 200 rupees"
     order_task -> "place an order for Ali of two mango lassi"

You are also given the conversation so far. Use it to fill in whatever the latest
message leaves out. "make it three instead" means nothing on its own, but the
history tells you which dish and which customer. Always write each task as a
complete instruction that stands on its own - never "the same as before", because
the workers cannot see the history.

Leave a task empty when that worker is not involved.
Keep the reason to one short sentence."""


class Route(BaseModel):
    """Which worker handles a request, and the part of the request each one gets."""

    worker: Literal["menu", "order", "both"] = Field(description="who handles this")
    reason: str = Field(description="a few words on why")
    menu_task: str = Field(default="", description="the menu worker's part, or empty")
    order_task: str = Field(default="", description="the order worker's part, or empty")


class State(TypedDict):
    request: str
    history: str  
    decision: str
    reason: str
    menu_task: str
    order_task: str
    menu_result: str
    order_result: str


def supervisor_node(state: State) -> dict:
    llm = make_llm(SUPERVISOR_MODEL).with_structured_output(Route)

    history = state.get("history", "")
    if history:
        question = f"Conversation so far:\n{history}\n\nLatest message:\n{state['request']}"
    else:
        question = state["request"]

    route = llm.invoke(
        [
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": question},
        ]
    )
    print(f"ROUTING: {route.worker} - {route.reason}")

    return {
        "decision": route.worker,
        "reason": route.reason,
        "menu_task": route.menu_task or state["request"],
        "order_task": route.order_task or state["request"],
    }


def menu_node(state: State) -> dict:
    return {"menu_result": run_agent(MENU_AGENT, state["menu_task"], "menu")}


def order_node(state: State) -> dict:
    return {"order_result": run_agent(ORDER_AGENT, state["order_task"], "order")}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("menu", menu_node)
    graph.add_node("order", order_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda s: "menu" if s["decision"] in ("menu", "both") else "order",
    )

    graph.add_conditional_edges(
        "menu",
        lambda s: "order" if s["decision"] == "both" else END,
    )

    graph.add_edge("order", END)

    return graph.compile()


