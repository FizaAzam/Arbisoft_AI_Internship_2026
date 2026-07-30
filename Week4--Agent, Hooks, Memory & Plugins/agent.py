import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from langgraph.checkpoint.memory import InMemorySaver
checkpointer=InMemorySaver()

from datetime import datetime
from langchain.agents.middleware import wrap_tool_call, ToolCallLimitMiddleware

from tools import search_web, read_file


model = ChatOpenAI(
    model = "openrouter/free",
    base_url = "https://openrouter.ai/api/v1",
    api_key = os.getenv("OPENROUTER_API_KEY")
)

@wrap_tool_call
def log_tool_calls(request, handler):
    
    start = datetime.now()
    print(f"[{start}] CALLING {request.tool_call['name']}  args={request.tool_call['args']}")

    result = handler(request)

    end = datetime.now()
    print(f"[{end}] DONE {request.tool_call['name']}")

    return result


limiter = ToolCallLimitMiddleware(tool_name="read_file",run_limit=4)
limiter= ToolCallLimitMiddleware(tool_name="web_search", run_limit=10)


agent= create_agent( 
    model =model,
    tools=[search_web, read_file],
    system_prompt = "You are a research assistant.\n"
    "- Search the web before answering anything time-sensitive, factual, or "
    "about current people, prices, products, or events.\n"
    "- You may ONLY cite URLs that appear in a tool result. Never write a URL "
    "from memory. If you did not call a tool, say so plainly and do not "
    "include any links.\n"
    "- Do not issue the same search query twice.\n"
    "- Do not put a year in your search query unless the user explicitly asked "
    "about a specific year.",
    checkpointer= checkpointer,
    middleware = [log_tool_calls, limiter]
    )


while True:
    user_input= input("\nYou: ")
    if user_input.lower() in ("exit", "quit", "bye"):
        break

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]}, 
        config={"configurable": {"thread_id": "session-1"}}
    )

    print(result["messages"][-1].content)




