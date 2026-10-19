#Run: .venv\\Scripts\\uvicorn.exe api:app --reload
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import db
import orchestrator

SESSIONS: dict[str, list[str]] = {}
MAX_HISTORY_LINES = 8

# ------------------------------------------------------- request / response
class ChatRequest(BaseModel):
    """What the browser sends us."""

    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = None  


class ChatResponse(BaseModel):
    """What API sends back."""

    reply: str
    session_id: str
    routed_to: str  
# ------------------------------------------------------------------- app

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the agents once at startup instead of on every request."""
    db.setup()  
    orchestrator.MENU_AGENT, orchestrator.ORDER_AGENT = orchestrator.build_workers()
    app.state.graph = orchestrator.build_graph()
    yield


app = FastAPI(title="Restaurant Chatbot", lifespan=lifespan)

#React runs on a different port, so the browser needs permission to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Cheap check that the server is up."""
    return {"status": "ok", "sessions": len(SESSIONS)}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """One turn of the conversation."""
    session_id = request.session_id or str(uuid.uuid4())
    history = SESSIONS.setdefault(session_id, [])

    try:
        final = app.state.graph.invoke({
            "request": request.message,
            "history": "\n".join(history[-MAX_HISTORY_LINES:]),
        })
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"The assistant is unavailable right now: {error}",
        )

    parts = [final.get("menu_result", ""), final.get("order_result", "")]
    reply = "\n\n".join(part for part in parts if part)

    history.append(f"customer: {request.message}")
    history.append(f"assistant: {reply}")

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        routed_to=final.get("decision", "unknown"),
    )

