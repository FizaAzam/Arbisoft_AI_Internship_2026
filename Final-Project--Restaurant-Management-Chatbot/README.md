# Restaurant Management Chatbot

A restaurant management system driven by conversation. A customer types an
ordinary sentence — *"add mango shake for 250, then order two for Ali"* — and a
supervisor agent routes it to the right specialist agent, which calls tools that
read and write a MySQL database.

Built for the Arbisoft AI Internship 2026.

## Architecture

```
React (localhost:5173)
   |  POST /chat  { message, session_id }
FastAPI  api.py               HTTP layer + per-session conversation history
   |  graph.invoke({ request, history })
LangGraph supervisor  orchestrator.py
   |         \
Menu agent    Order agent     each is given only its own tools
   |         /
Tools  restaurant.py          view_menu, add_menu_item, place_order, view_orders
   |
db.py  ->  MySQL restaurant_db
```

### How the agents coordinate

The supervisor reads each message and returns a structured decision: `menu`,
`order`, or `both`, together with a self-contained instruction for each worker it
picks. On `both` the menu agent runs first, so a dish that was just added can be
ordered in the same turn.

Separation is enforced by the tool lists at the bottom of `restaurant.py`, not by
prompting. The menu agent has no ordering tool, so it cannot place an order even
if asked to.

Only the supervisor is shown the conversation history. Its job is to rewrite
references like *"make it three instead"* into complete instructions, so the
workers stay stateless.

## Files

| File | Purpose |
|---|---|
| `config.py` | Reads settings and the database password from `.env` |
| `db.py` | Opens MySQL connections; `setup()` creates the schema |
| `restaurant.py` | The four tools and the SQL behind them |
| `orchestrator.py` | Supervisor, the two agents, and the LangGraph graph |
| `api.py` | FastAPI `/chat` endpoint and session history |
| `frontend/` | React chat interface |

## Setup

Requires Python 3.11+, Node.js 18+, and a running MySQL server.

### Backend

```powershell
uv venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
PASSWORD=your_mysql_password
OPENROUTER_API_KEY=your_openrouter_key
```

The database and its tables are created automatically on first start.

### Frontend

```powershell
cd frontend
npm install
```

## Running

Two terminals.

**Backend** (port 8000):
```powershell
.venv\Scripts\python.exe -m uvicorn api:app --reload
```

**Frontend** (port 5173):
```powershell
cd frontend
npm run dev
```

Then open http://localhost:5173

### Without the frontend

- Swagger UI at http://127.0.0.1:8000/docs sends requests to `/chat` directly.
- `.venv\Scripts\python.exe orchestrator.py` runs five demo requests through the
  orchestration layer alone, printing each routing decision.

## Try it

```
what do you have on the menu?
add mango shake for 250
I'm Ali and I'd like two chicken biryanis
actually make it three instead
add gulab jamun for 150, then order one for Sara
what orders have been placed?
```

## Design notes

**The agents never write SQL.** They choose which function to call and with what
arguments; the SQL is ordinary Python that the model never sees.

**Tool errors are written for the model to read.** Ordering a dish that is not on
the menu returns an error containing the current menu, so the agent can correct
itself and offer the customer real options.

**`place_order` validates before it writes.** Every dish name and quantity is
checked first, so one bad item cannot leave a half-created order in the database.

**`setup()` is not a tool.** No customer message can reach it, so no conversation
can create or alter tables.

## Known limitations

- Sessions are held in memory and are lost when the server restarts.
- There is no update-order tool, so changing an order records a new one.
- Free OpenRouter models are rate-limited; requests occasionally return 502.
