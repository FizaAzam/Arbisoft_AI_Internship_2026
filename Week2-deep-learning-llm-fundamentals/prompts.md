# Prompts Log 

A record of the prompts I used while building the CLI chat app and the model
comparison, plus the prompting techniques from my notes.

---

## 1. Model comparison prompt
Used in `compare_models.py` — sent to all 3 models unchanged so the comparison
would be fair.

```
Explain what an API is to a 12-year-old in 3 sentences.
```

**Why this prompt:** It is simple, has a clear audience ("a 12-year-old") and a
clear constraint ("3 sentences"), which makes it easy to compare how different
models explain the same idea.

---

## 2. System prompt (CLI chat app)
Used in `chat.py` as the permanent instruction that defines the assistant's role.

```
You are a helpful assistant.
```

**Why:** A system prompt is like giving the AI a permanent job before the
conversation starts. Every reply is shaped by it.

---

## Prompting techniques (from notes)

- **Zero-shot** — give instructions with no examples (what my prompts above use).
- **One-shot** — give one example of the desired output.
- **Few-shot** — give several examples so the model copies the pattern
  (costs more tokens, but improves consistency).
- **Chain of Thought (CoT)** — ask the model to reason step-by-step before
  answering, instead of jumping straight to the answer.

---

