# Prompts Log 

A record of the prompts I used while building the CLI chat app and the model
comparison, with the answers condensed.

**Tasks:** A terminal chat app with conversation memory · Comparing three free models on the same prompt
**Stack:** `openai` library pointed at OpenRouter · `python-dotenv` · free `:free` models

---

## 1. Connecting to an LLM from Python

**How do I call an LLM from a Python script?**

The `openai` library. Create a client, then `client.chat.completions.create(...)`
with a model name and a list of messages.

**do I need a different library?**

No. Point the same client at a different `base_url`:

```python
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
```

OpenRouter speaks the same request/response format as OpenAI, so the library works
unchanged. Only the URL, the key, and the model string differ.

**How do I keep my API key out of the code?**

Put it in a `.env` file, call `load_dotenv()` at the top, and read it with
`os.getenv("OPENROUTER_API_KEY")`. The key never appears in a file that could be
committed.

**Why is the reply in `response.choices[0].message.content`?**

The API can return several alternative completions, so `choices` is a list —
`[0]` takes the first. `.message` is the message object and `.content` is its text.

---

## 2. Building the CLI chat app

| Prompt | Answer |
|---|---|
| How do I give the model a permanent instruction/role for the whole conversation? | A **system message** as the first entry in the `messages` list. It's a standing instruction that shapes every reply. |
| How does the model remember what was said earlier? | It doesn't — I do. Every turn is a fresh API call, so the whole `messages` list gets re-sent each time. Memory means not throwing that list away. |
| What are the three roles for? | `system` = the standing instruction, `user` = what I type, `assistant` = what the model replied. |
| Why append the model's own reply back into `messages`? | Otherwise the next call would only contain my questions, and the model would have no record of its own answers — it would lose half the conversation. |
| How do I let the user exit cleanly? | `if user_input.lower() in ("quit", "exit"): break` before the API call, so typing "quit" doesn't get sent to the model. |

**The loop, in short:** read input → append as `user` → send the whole list → print
the reply → append it as `assistant` → repeat.

---

## 3. Prompts used in the code

### System prompt (`chat.py`)

```
You are a helpful assistant.
```

**Why:** A system prompt is a permanent job description given to the model before
the conversation starts. This one is deliberately plain.

### Comparison prompt (`compare_models.py`)

```
Explain what an API is to a 12-year-old in 3 sentences.
```

**Why this prompt:** It has a clear audience (*"a 12-year-old"*) and a clear
constraint (*"3 sentences"*), which makes it easy to compare how different models
explain the same idea. Sent to all three models **unchanged** so the comparison
stays fair.

---

## 4. Comparing the models fairly

| Prompt | Answer |
|---|---|
| How do I make sure the comparison is actually fair? | Change exactly one thing — the model name. Same prompt, same client, same call shape, so any difference in output is the model's doing. |
| Why send no system prompt or history in the comparison script? | Each model gets a clean single-turn call. Adding a system prompt or prior context would be another variable to control. |
| How do I measure speed? | `time.time()` before and after the call; `elapsed = time.time() - start`. |
| Free models keep failing mid-run — how do I stop one bad model killing the whole comparison? | Wrap each call in `try/except` and print `"[This model is unavailable right now: {e}]"`, so the loop continues to the next model instead of crashing. |

---

## 5. Results

**Prompt (identical for all models):** *"Explain what an API is to a 12-year-old in 3 sentences."*
**Date:** 2026-07-16 · **Tool:** OpenRouter free-tier models via `compare_models.py`

> Results were collected across several runs because the free models are frequently
> rate-limited (HTTP 429). The prompt was identical every time, so the comparison
> is still fair.

| Model | Speed | Quality of answer | Best use-case fit |
|---|---|---|---|
| `openai/gpt-oss-20b:free` | Slow and variable (~7–21s; once returned an empty response) | Correct and clear ("restaurant menu / waiter" analogy), but inconsistent on the free tier | Reliable wording when it works; not ideal when speed matters |
| `google/gemma-4-31b-it:free` | Fastest (~3s) | Concise and correct; "messenger / digital waiter" analogy plus a concrete weather-app example | Quick answers, simple Q&A, chatbots |
| `nvidia/nemotron-3-super-120b-a12b:free` | Fast (~2.5–5s) | Correct and the most detailed; "waiter in a restaurant" analogy with a game and weather example | Richer, more thorough explanations |

**Speed.** Gemma and Nemotron were consistently fast (~3–5s). gpt-oss was slowest
and most unpredictable — from ~7 to ~21 seconds, and once "thought" for 33 seconds
and returned an empty (`None`) answer.

**Quality.** All three used the classic restaurant analogy correctly. Nemotron gave
the richest answer, Gemma the most concise. gpt-oss was good when it worked.

**Use-case fit.** Fast and simple → Gemma. Detailed → Nemotron. gpt-oss → too slow
and inconsistent on the free tier for anything time-sensitive.

---

## 6. What I learned

| Prompt | Answer |
|---|---|
| Why did one model return an empty answer instead of text? | gpt-oss is a reasoning-style model — on a busy free provider its output can get cut off, leaving `content` as `None`. Worth knowing because appending a `None` reply back into `messages` would corrupt the history. |
| Why do free models keep returning 429? | Rate limiting. A real app needs **error handling and retry logic**, not a single unguarded API call. |
| What made swapping models so easy? | Only the model string changes — the client, the request shape and the response parsing all stay identical. That's the practical benefit of every provider speaking the same API format. |

