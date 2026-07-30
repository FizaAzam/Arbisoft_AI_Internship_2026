# Prompts Log

A record of the prompts I used while building the research agent, with the answers
condensed. Grouped by topic rather than in the order asked.

**Task:** Build a research agent with a web-search skill using LangChain.
**Stack:** LangChain v1 (`create_agent`) · OpenRouter (free models, OpenAI-compatible API) · SerpAPI (Google search) · pdfplumber (PDF reading)

---

## 1. Getting oriented before writing any code

**explain the overall concept — what actually makes something a "research agent"?**

Four things: `model + tools + a loop + a system prompt`. A plain LLM call is one
request, one response. An agent wraps the same model in a loop that lets it *do
things* between the question and the answer — decide it needs information, fetch
it, look at what came back, decide whether it needs more.

**How are tools executed?**

No — and this is the key idea. The LLM cannot execute anything. It emits a
structured request saying *"call `search_web` with `query='...'`"*. My code receives
that, runs the real function, and hands the result back. Everything that actually
happens, happens in my program.

**What does the loop look like?**

```
send messages to the model
      ↓
did the response contain tool calls?
      ↓ no  → stop, return the answer
      ↓ yes → run each tool, append the result as a ToolMessage, repeat
```

Each pass the model sees everything before it, which is what lets it chain steps.

**What goes into making a tool function?**

1. An ordinary Python function (build params, call the API, parse JSON, format a string).
2. Type hints — they become the parameter types in the schema the model sees.
3. A docstring **written for the model**, not for a human — this is what it reads to decide if the tool is relevant.
4. Return a short string — whatever it returns enters the model's context, so trim at the tool boundary.
5. Return errors as strings instead of crashing, so the model can react to them.
6. Decorate with `@tool` to wrap it in a `StructuredTool` carrying the metadata the agent needs.

**Which libraries do I need and what is each for?**

| Library | Role |
|---|---|
| `langchain` | The framework — `create_agent` (the loop), `@tool`, middleware hooks |
| `langchain-openai` | Client for any OpenAI-compatible endpoint |
| `langgraph` | The checkpointer (`InMemorySaver`) that gives the agent memory |
| `requests` | HTTP calls inside the tool |
| `python-dotenv` | Loads API keys from `.env` |
| `pdfplumber` | Extracts text from PDFs |

**How does the program connect to the LLM?**

Through an OpenAI-compatible endpoint — a **wire format**, not the company. Many
providers speak it, so the same client reaches any of them by changing `base_url`:

```python
model = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
```

**I have two API keys — what does each do?**

OpenRouter key talks to the LLM (the reasoning). SerpAPI key performs the web
search (the information). Both live in `.env`, loaded with `load_dotenv()`.

---

## 2. Building the search tool

| Prompt | Answer |
|---|---|
| How do I call a web API from Python? | The `requests` library — `requests.get(url, params=..., timeout=...)`. |
| How do I make a failed HTTP response raise an error instead of continuing silently? | `response.raise_for_status()` — raises on 404, 500, etc., so control jumps to `except`. |
| How do I turn the response body into something Python can use? | `response.json()` parses the JSON body into a dict. |
| What is the `url` in `requests.get()` for? | The endpoint being called — `https://serpapi.com/search`. |
| Is SerpAPI a search engine? | No. It's a middleman that queries other engines and returns clean JSON — hence `"engine": "google"` in `params`. |
| What is Brave then? | Brave runs its **own** index. Its key reaches Brave's results directly; SerpAPI proxies whichever engine you name. |
| How do I ask for a specific number of results? | The `num` parameter — but Google ignores it. Upstream params are advisory. |
| So how do I actually limit it to 5? | Slice client-side: `data.get("organic_results", [])[:5]`. Writing `[:5]` *inside* `.get()` fails — a slice isn't a value. |
| How can there be an error inside a response that succeeded? | Two layers: the **HTTP layer** (status code) and the **application layer** (an `error` key in the body). A request can succeed at one and fail at the other. |
| When do I use `if` vs `try/except`? | `try/except` for things that **throw**; `if` for conditions you can **inspect**. |
| How do I check the results came back empty? | `if not results:` — true for an empty list. |
| How do I return one string when I built a list of pieces? | `"\n\n".join(formatted)`. |

---

## 3. Turning the function into a tool

| Prompt | Answer |
|---|---|
| How do I turn my plain function into something the agent can use? | `@tool`. It wraps it in a `StructuredTool` with `.name`, `.description`, `.args_schema`, `.invoke()`. |
| Why convert it to an object instead of just calling the function? | The agent needs that metadata to describe the tool to the model and to look it up by name when the model asks for it. |
| How does the model know what my tool does? | The **docstring**. For a normal function it's human documentation; for a tool it's what the model reads to decide relevance. |
| How do I give each parameter its own description? | `@tool(parse_docstring=True)` — splits the `Args:` block out instead of leaving one flat description. |
| Why did my `Args:` block disappear? | I split the docstring into two string literals. Only the **first** becomes `__doc__`. |
| Why did `query description:` raise a `ValueError`? | The name before the colon must **exactly match** a parameter in the signature. |
| How can I see exactly what the model receives? | `convert_to_openai_tool(search_web)` prints the JSON schema sent to the LLM. |

---

## 4. Building the agent

| Prompt | Answer |
|---|---|
| How do I make the model actually run my tool instead of just asking for it? | `create_agent(model=..., tools=[...])`. Passing `tools` gives it something to **execute**, which is what enables the loop. |
| Is the only difference from a normal LLM call that it has a `tools` parameter? | The `tools` parameter is what triggers the loop — and the loop is what makes it an agent. |
| So `create_agent` has a loop inside it? | Yes: call model → no `tool_calls`, stop → otherwise run each tool, append a `ToolMessage`, repeat. |
| How is it decided whether a tool gets called? | The model decides. It sees the tool schema on **every** request and predicts either text or a tool call. It's a **prediction, not a decision tree** — the same question can go either way. |
| How do I see everything that happened, not just the answer? | `result["messages"]` — `HumanMessage` → `AIMessage` (with `.tool_calls`) → `ToolMessage` → `AIMessage`. |
| How can I print the whole conversation with readable formatting? | `m.pretty_print()` on each message — it produces the `==== Ai Message ====` banners and shows tool calls and args. |
| How do I get just the final answer? | `result["messages"][-1].content`. |

---

## 5. Providers and APIs

| Prompt | Answer |
|---|---|
| How do I use a provider other than OpenAI with `langchain-openai`? | Change `base_url`. OpenRouter, Groq and Together all speak the same wire format. |
| If I switch the search backend to Brave, does only the key change? | No. Different URL, the key moves into a **request header**, `engine` is dropped, and the response JSON has a different shape (`data["web"]["results"]`). Only the skeleton stays. |
| I built this against an OpenAI-compatible API — what differs with the Claude API? | Four things: |

| | OpenAI-compatible (used here) | Claude API |
|---|---|---|
| Tool schema | `{"type": "function", "function": {...}}` | Flat: `{"name", "description", "input_schema"}` |
| Tool call in response | Separate `tool_calls` field | A `tool_use` **content block** |
| Arguments | A **JSON string** — needs `json.loads()` | An **already-parsed dict** |
| Sending results back | `role: "tool"` + `tool_call_id` | `role: "user"` with a `tool_result` block |

**What would change if I swap to Claude?** Message objects and my loop stay
identical — LangChain normalises both. `response_metadata` keys differ
(`stop_reason` vs `finish_reason`), and `.content` can be a **list of blocks**
instead of a string.

---

## 6. Adding memory

| Prompt | Answer |
|---|---|
| How do I make the agent recall facts from earlier in the session? | Stop discarding the message list. The model has **no memory between calls** — memory means re-sending the conversation. Manually: keep `messages` outside the loop, append each question, then `messages = result["messages"]`. |
| How do I hand that job to LangChain instead? | A **checkpointer**: `create_agent(..., checkpointer=InMemorySaver())`. It saves state after every `invoke()` and reloads it before the next. |
| How does it know which conversation a message belongs to? | The `thread_id` in `config={"configurable": {"thread_id": "session-1"}}` — it works like a dictionary key. |
| Is this program one chat thread? | Yes — `thread_id` is hardcoded. And `InMemorySaver` stores in RAM, so history dies when the process exits even though the label is reused. |

---

## 7. Hooks and middleware

| Prompt | Answer |
|---|---|
| How do I log every tool call with timestamps? | A `@wrap_tool_call` middleware function, passed as `create_agent(..., middleware=[log_tool_calls])`. |
| What's the difference between `@tool` and `@wrap_tool_call`? | `@tool` adds a capability the **model can choose** — its docstring is written *for the model*. `@wrap_tool_call` is **invisible to the model** and runs around every tool call automatically. |
| Are `request` and `handler` passed automatically? | Yes — `create_agent` supplies both. `request` is what the model asked for; calling `handler(request)` runs the real tool. |
| How do I get code to run before *and* after the tool? | Put it either side of `handler(request)`. Because I control when it's called, everything before is "pre" and everything after is "post" — and variables stay in scope across both. |
| What is `query` in `request.tool_call['args']`? | The parameter name I chose in `def search_web(query: str)`. The dict holds what the model filled in. |
| How does LangChain know if a hook is pre or post? | The **decorator name** declares it — `@before_model`, `@after_model`, `@wrap_model_call`, `@wrap_tool_call`. Function names are irrelevant. |
| Can `@tool` limit how many times a tool runs? | No. That's `ToolCallLimitMiddleware`, a **pre-built class** you instantiate (no decorator). `run_limit` = per prompt, `thread_limit` = per session. |
| When does that limiter run? | It hooks `after_model` — after the model produces tool calls (something to count) but before they execute (still blockable). |
| Would `run_limit=1` break multi-hop? | Largely yes. Scoped to `search_web` it still allows one `read_file` + one search, but any hop needing two searches gets cut off. It's a guard against runaway loops, not something to set tight during a multi-hop demo. |

---

## 8. Adding the file-read plugin

| Prompt | Answer |
|---|---|
| How do I read a PDF in Python? | `pdfplumber.open(path)`, then loop `pdf.pages` calling `.extract_text()`. It can return `None` for pages with no extractable text, hence `or ""`. |
| Doesn't LangChain have its own PDF reader? | Yes, `PyPDFLoader` — but it wraps `pypdf`, pulls in the large `langchain_community` package, and returns `Document` objects you still have to unwrap. Using `pdfplumber` directly was more proportionate. |
| Why `join` the pages? | Each page produces its own string; the function must return **one** string. |
| Explain `os.path.splitext(path)[1].lower()` | `splitext` = "split extension". Returns `(root, ext)`; `[1]` takes the extension; `.lower()` makes the check case-insensitive. |
| How do I stop a missing file from crashing the agent? | Check `os.path.exists(path)` first and **return** an error string instead of letting the exception propagate. |
| Should the docstring say "check all local files when you don't have an answer"? | No — the tool takes one `path` and can't list a directory, and a vague fallback would have it reading unrelated project files. That would need a separate `list_files` tool. |
| So it won't work unless I name the file? | Correct. The model can't browse the folder. Confirmed in testing: *"use the file i gave you"* failed; typing `testingnote.txt` worked immediately. |

---

## 9. Reasoning patterns

| Prompt | Answer |
|---|---|
| How do I incorporate chain of thought — through the system prompt? | Yes. CoT is a **prompting technique**, not an API parameter. |
| Is adding that to the system prompt really *implementing* CoT? | Yes — that's **zero-shot CoT**, the complete technique. |
| Are there types of chain of thought? | Zero-shot, few-shot, self-consistency (sample several chains, take the majority), least-to-most (decompose into sub-questions), tree-of-thought (branching paths with backtracking). |
| Explain few-shot CoT | Examples in the prompt that show the *reasoning steps*, not just correct answers, so the model imitates the pattern rather than inferring it from an instruction. |
| What is the ReAct pattern? | **Reasoning + Acting** — reason about what's needed, call a tool, observe the result, reason again, loop until able to answer. |
| Can I implement it? Does changing the system prompt count? | **No** — unlike CoT, the ReAct loop was already implemented by `create_agent`. A prompt change only makes the Thought step **visible as text**. |
| Where does each part of ReAct live in my code? | Thought = the model's decision inside `model.invoke()`. Action = `response.tool_calls`. Observation = the `ToolMessage` appended back. Repeat = the loop inside `create_agent`. |

---

## 10. Fixing the prompt after a real failure

**Should I refine the system prompt to fix what went wrong in hop 2?**

Yes — the failure was the prompt being too loosely worded.

**Version 1:**

```
You are a research assistant. You should search before answering and you must
cite the cources you use to answer.
```

**What went wrong (2026-07-29).** For *"Who is the current CEO of the company that
makes the latest iPhone, and what have they said recently about AI?"* the agent
answered in full detail with a Sources section linking to The Verge and CNBC — but
the tool log showed **no `CALLING` / `DONE` lines at all**. `search_web` was never
invoked. The model answered from stale training data and **invented the URLs**.

**Why version 1 allowed it.** It said *"cite the sources you use"* — the model
complied, it produced citations. Nothing required them to **come from a tool
result**, so inventing plausible URLs satisfied the rule.

A second issue: the model put a hardcoded year in its own query
(`"latest OpenAI model 2025"`), biasing results toward old news. It reported GPT-5
as "the latest" when an earlier run without a year had correctly found GPT-5.6.

**Version 2:**

```
You are a research assistant.
- Search the web before answering anything time-sensitive, factual, or about
  current people, prices, products, or events.
- You may ONLY cite URLs that appear in a tool result. Never write a URL from
  memory. If you did not call a tool, say so plainly and do not include any links.
- Do not issue the same search query twice.
- Do not put a year in your search query unless the user explicitly asked about a
  specific year.
```

| Rule added | Failure it targets |
|---|---|
| Explicit list of what is search-worthy | Skipping search on a question it assumed it knew |
| "ONLY cite URLs that appear in a tool result" | Fabricated citations — turns a soft instruction into a testable prohibition |
| "If you did not call a tool, say so plainly" | Gives a compliant way to answer without inventing sources |
| "Do not issue the same search query twice" | Duplicate identical `search_web` calls |
| "Do not put a year in your search query…" | Self-inflicted staleness from the model's own wording |

**Will this guarantee it?** No — these are instructions, not guarantees. The
identical prompt produced a search in one session and none in another. Anything
that must hold every time belongs in code: a caching `@wrap_tool_call` middleware
for duplicate calls, `ToolCallLimitMiddleware(run_limit=N)` for runaway loops.

**The prompt should nudge; the middleware should enforce.**

---

## 11. Testing the agent

**Give me prompts that would definitely need a web search.**

Anything with *latest, current, right now, today*, or a fast-changing number —
these match the `search_web` docstring almost word for word. A useful contrast for
testing the boundary: *"what is inflation?"* (conceptual, may not search) vs
*"what is Pakistan's inflation rate this month?"* (specific and current, should search).

**Should I test with multi-hop questions?**

Yes — multi-hop means a question that can't be answered in one lookup, where each
step depends on the previous result.

| Test prompt | Outcome |
|---|---|
| "Read testingnote.txt to find my favorite color, then search the web for what that color symbolizes in different cultures." | PASS — `read_file` returned "teal", then `search_web("teal color symbolism in different cultures")`. The second query was genuinely derived from the first. |
| "Who is the current CEO of the company that makes the latest iPhone, and what have they said recently about AI?" | FAIL — no tool call, fabricated citations. See §10. |
| "What's the latest OpenAI model?" | PARTIAL — answered, but issued the same query twice and hardcoded "2025", retrieving stale results. |
| "Now find out if the company behind the model you just mentioned has said anything about it publicly this week." | PASS — resolved "the model you just mentioned" from session memory, then searched on that basis. |
| "What's the secret code word in notes.txt?" | PASS — the file held something unguessable, so a correct answer proved the tool actually ran. |

Two of four multi-hop tests behaved as intended. The two failures are recorded as
findings, not fixed-and-forgotten bugs.

---

## 12. Why free models kept failing

**Three different models failed in a row — is this my code?**

No, all three were provider-side: NVIDIA 502 (`ResourceExhausted`), Google AI Studio
429 (rate-limited upstream), and an empty final message from `gpt-oss-20b:free`
(reasoning-channel quirk — `reasoning_tokens: 299`, `content: ""`). Fixed by
switching to `openrouter/free`, which auto-routes to an available model.

Other observations worth recording:

- **Tool triggering is non-deterministic.** The identical prompt (*"what's the latest OpenAI model?"*) searched in one run and not the next — producing a correct 2026 answer versus a confidently wrong 2025 one.
- **Instruction-following varies by model.** Some free models cited sources properly; others ignored the instruction or fabricated links.
- **Phrasing decides tool use.** *"use the file i gave you"* did not trigger `read_file`; the literal filename did.
