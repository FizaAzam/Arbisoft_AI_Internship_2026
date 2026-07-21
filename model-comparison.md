# Model Comparison (via OpenRouter)

**Prompt used (identical for all models):**
> "Explain what an API is to a 12-year-old in 3 sentences."

**Date:** 2026-07-16
**Tool:** OpenRouter API (free-tier models), called from `compare_models.py`

> Note: Results were collected across several runs because the free models are
> frequently rate-limited (HTTP 429 errors). The prompt was identical every time,
> so the comparison is still fair.

---

## Results

| Model | Speed (time taken) | Quality of answer | Best use-case fit |
|-------|--------------------|-------------------|-------------------|
| `openai/gpt-oss-20b:free` | Slow & variable (~7–21s; once returned an empty response) | Correct and clear ("restaurant menu / waiter" analogy), but inconsistent on the free tier | Reliable wording when it works; not ideal when speed matters |
| `google/gemma-4-31b-it:free` | Fastest (~3s) | Concise and correct; used a "messenger / digital waiter" analogy plus a concrete weather-app example | Quick answers, simple Q&A, chatbots |
| `nvidia/nemotron-3-super-120b-a12b:free` | Fast (~2.5–5s) | Correct and the most detailed; "waiter in a restaurant" analogy with a game + weather example | Richer, more thorough explanations |

---

## Notes

**Speed**
- Gemma and Nemotron were consistently fast (~3–5 seconds).
- gpt-oss was the slowest and most unpredictable — it ranged from ~7 to ~21 seconds,
  and once "thought" for 33 seconds and returned an empty (`None`) answer. This is
  because gpt-oss is a reasoning-style model, so on a busy free provider its output
  can get cut off.

**Quality**
- All three explained the concept correctly using the classic "restaurant" analogy.
- Nemotron gave the richest answer (extra real-world examples).
- Gemma gave the most concise, easy-to-read answer.
- gpt-oss was good when it worked, but unreliable on the free tier.

**Use-case fit**
- Need it fast and simple → **Gemma**.
- Need a detailed, well-explained answer → **Nemotron**.
- gpt-oss → fine for general use, but too slow/inconsistent on free tier for
  anything time-sensitive.

---

## What I learned
- Free models on OpenRouter often return `429` (rate-limited) errors, so a real
  app needs **error handling and retry logic**, not just a single API call.
- Reasoning models (like gpt-oss) can be slow and sometimes return empty responses.
- OpenRouter makes comparison easy: only the model name changes in the code —
  everything else stays the same.
