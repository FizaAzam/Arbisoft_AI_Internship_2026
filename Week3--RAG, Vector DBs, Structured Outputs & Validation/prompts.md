# Prompts Log 

A record of the prompts I used while building the RAG demo, the structured-output
pipeline, the LangChain rebuild, and the embedding-model comparison - with the
answers condensed.

**Tasks:** Chat with your PDFs (RAG) · Structured JSON output with validation · The same pipeline rebuilt in LangChain · Comparing two embedding models on retrieval quality

**Stack:** `pdfplumber` · `langchain-text-splitters` · `sentence-transformers` (local embeddings) · `chromadb` · OpenRouter (free model) · `pydantic` v2 · `pytest`

---

## 1. Understanding RAG before building it

**Walk me through what RAG actually does end to end.**

Retrieve the relevant passages from *my* documents, then let the LLM answer *using
those passages*. The answer is grounded in real source text instead of the model's
memory, which is what cuts hallucination.

```
ingest.py  (run once)                  query.py  (run per question)
─────────────────────                  ────────────────────────────
1. Extract text   (pdfplumber)         5. Retrieve      (Chroma similarity search)
2. Chunk          (LangChain splitter) 6. Build context (+ [source, page] labels)
3. Embed          (MiniLM, local)      7. Generate      (LLM via OpenRouter)
4. Store          (ChromaDB on disk)
```

**Why chunk at all instead of embedding whole pages?**

Whole pages produce blurry embeddings — too many topics averaged into one vector.
Small focused chunks produce sharp ones, so retrieval matches the question
precisely, and only relevant text gets sent to the LLM.

**I'm using recursive fixed-size splitting — what would semantic chunking do differently?**

Fixed-size cuts at character counts. Semantic chunking cuts where the *meaning*
shifts, so a chunk boundary lands at a topic change rather than mid-argument.

**Is the sentence-transformer model what generates the embeddings locally?**

Yes — `all-MiniLM-L6-v2`, run through `sentence-transformers`, turns text into
vectors on my own machine.

**What do I actually gain and lose by embedding locally rather than through an API?**

Gain: free, offline after one download, private (text never leaves the machine).
Lose: bounded by my hardware, and a smaller model than a hosted one.

**What does ChromaDB give me that a normal database doesn't?**

It stores embeddings alongside the text and metadata, and searches by **meaning**
(nearest vectors) rather than exact keywords.

**How is the data organised inside Chroma?**

In a **collection** — effectively a table where each row is one chunk: id + text +
embedding + metadata. `collection.count()` returns the number of stored chunks.

**How do I avoid re-embedding every time I run a query?**

`PersistentClient()` — opens or creates a ChromaDB that lives on disk, so I embed
once during ingest and reuse the vectors on every query afterwards.

**When is RAG actually worth it over just pasting a PDF into ChatGPT?**

Pasting works for one small document. RAG scales to many or large documents, costs
less (only the relevant chunks get sent, not the whole corpus), and persists as a
reusable knowledge base.

---

## 2. Building the ingest and query scripts

| Prompt | Answer |
|---|---|
| Why sort the results of `glob("*.pdf")`? | Guarantees the PDFs are read in the same order every run, so chunk IDs stay stable and re-ingesting overwrites cleanly instead of duplicating. |
| What actually happens when I call `collection.query()`? | It embeds the question with the **same** model used at ingest, compares that vector against every stored one, and returns the closest `k` chunks ranked by distance. |
| Why is every field in `results` a list of lists — why `["documents"][0]`? | Chroma supports querying several questions at once, so the outer list has one slot per question. `[0]` unwraps my single question's matches. |
| How do I walk `documents` / `metadatas` / `distances` together? | `zip()` — it iterates the parallel lists in step so each chunk's text, metadata and distance get bundled into one `hit`. |
| How do I turn the retrieved chunks into something the LLM can cite from? | `build_context()` flattens them into one labeled block — `[source, page]` followed by the text — which becomes the LLM's reference material. |
| How do I stop the model answering from its own knowledge instead of the context? | The system prompt. It's the standing instruction to answer ONLY from the provided context and cite sources — the guardrail that keeps answers grounded. |

**Configuration knobs I exposed:**

| Setting | File | Meaning |
|---|---|---|
| `CHUNK_SIZE` | `ingest.py` | max characters per chunk (default 1000) |
| `CHUNK_OVERLAP` | `ingest.py` | shared characters between neighbouring chunks (default 150) |
| `TOP_K` | `query.py` | how many chunks to retrieve per question (default 3) |
| `LLM_MODEL` | `query.py` | the OpenRouter model (a free `:free` slug) |

---

## 3. Structured output — making the LLM return validated JSON

**How do I make the LLM return structured JSON instead of prose, and actually trust it?**

Four parts: a schema that defines what valid looks like, a prompt that demands that
shape, and two gates that reject anything that doesn't match. The pipeline in one
line: `LLM → JSON → validate → save`.

**How do I define "valid" without writing manual checks?**

A Pydantic v2 model — the type hints **are** the validation rules:

```python
class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")     # reject invented fields
    answer: str
    source_files: list[str]
    confidence: Literal["high", "medium", "low"]  # only these 3 values
```

Verified by hand in the REPL first: a good payload returns an object; bad enum,
wrong type, and extra field each raise `ValidationError`.

**Why two separate gates rather than one check?**

They catch different failures. The LLM hands back a **string** that can't be trusted:

```
Gate 1 — json.loads(raw)                 → JSONDecodeError (code fences, prose, cut-off text)
Gate 2 — StructuredAnswer.model_validate → ValidationError (wrong type / field / enum)
```

Only if **both** pass does anything reach disk. A `Gate 0` was added later for the
LLM call itself failing.

**How do I reuse `query.py` without breaking its original behaviour?**

`generate_answer` was refactored to take the system prompt as an argument, so the
structured task can pass its JSON-demanding prompt while task 1's prose prompt
still works.

---

## 4. Testing the validator

| Prompt | Answer |
|---|---|
| How do I test validation without burning LLM calls? | Hand the validator hand-written malformed payloads directly. The tests never call the model. |
| A test using `pytest.raises(...)` passes when the code throws — isn't that backwards? | No — the gate is *supposed* to throw on bad data, so green means the validation works. |
| How do I import the module in tests without firing a real LLM call? | The `if __name__ == "__main__":` guard keeps importing side-effect-free. |

| Test | Gate | What breaks |
|---|---|---|
| `test_good_payload_passes` | 2 | nothing (control) |
| `test_missing_field_is_rejected` | 2 | required field absent |
| `test_wrong_type_is_rejected` | 2 | list expected, string given |
| `test_bad_enum_is_rejected` | 2 | confidence outside high/medium/low |
| `test_extra_field_is_rejected` | 2 | invented field (`extra="forbid"`) |
| `test_non_json_string_is_rejected` | 1 | not parseable JSON |
| `test_fenced_json_is_rejected_by_parser` | 1 | ```` ```json ```` fences break `json.loads` |
| `test_valid_json_string_flows_through_both_gates` | 1+2 | happy path, end to end |

Result: **8 passed**, without firing the LLM.

---

## 5. Hallucination experiments

**How do I test whether the guardrail prompt is actually doing anything?**

Run two out-of-context questions (capital of France, quicksort complexity — nothing
about either is in the RAG papers) plus one in-context question as a control, then
re-run with the guardrail line removed and compare.

**Finding 1 — the guardrail worked.** With *"If the answer is not in the context,
say you don't know and set confidence to low"* present, the model honestly refused
both out-of-context questions: `confidence: "low"`, `source_files: []`. No
hallucination.

**Finding 2 — removing that one line produced a fabricated citation.** The France
question returned:

```json
{
  "answer": "The provided context does not contain information about the capital of France...",
  "source_files": ["Evaluation of Retrieval-Augmented Generation.pdf",
                   "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf"],
  "confidence": "low"
}
```

It says *"I don't know"* yet cites two papers that contain nothing about France —
self-contradictory.

**Why didn't the validation gates catch that?**

Because the JSON is structurally perfect — a valid list of valid strings — so both
gates passed it. **Schema validation guarantees structure, not truth.** Catching
this needs a semantic check ("is every cited file actually one of the retrieved
hits?"), which Pydantic cannot express.

**Finding 3 — confidence drifted upward.** On the in-context control question,
removing the guardrail nudged `confidence` from `medium` to `high`. The model got
less cautious across the board.

| Question | Guardrail ON | Guardrail OFF |
|---|---|---|
| quicksort | refuses, `sources: []` | refuses, `sources: []` (still clean) |
| France | refuses, `sources: []` | refuses **but cites 2 papers** (fabricated citation) |
| RAG (control) | correct, `confidence: medium` | correct, `confidence: high` |

---

## 6. The citation-on-refusal failure, and the prompt fix

**The obvious out-of-context questions are too easy — how do I probe the gray zone?**

Use **in-domain** questions asking for a specific fact the papers cover but that
isn't in the top-3 retrieved chunks — e.g. *"How many passages are in the Wikipedia
index used by the RAG retriever?"* The strong guardrail stayed **on** throughout;
the prompt was never weakened.

**What went wrong.** Even with the guardrail on, the model said it didn't know and
still cited a source:

```json
{
  "answer": "The provided context does not mention the number of passages in a Wikipedia index used by a RAG retriever.",
  "source_files": ["Evaluation of Retrieval-Augmented Generation.pdf"],
  "confidence": "low"
}
```

The prompt defines `source_files` as *"the source filenames you **used**."* It used
nothing, yet listed a paper anyway. It was also inconsistent — other "don't know"
answers in the same run correctly returned `[]`.

**How was it caught?** By reading the output: the `answer` text directly
contradicts a non-empty `source_files`.

**How do I fix it in the prompt?** One line **added** (strengthening, not weakening):

```
The source_files list must contain ONLY files you actually used to answer. If you don't know
or the answer is not in the context, source_files MUST be an empty list [] and confidence MUST be low.
```

**Result on the same question:**

| | `source_files` |
|---|---|
| BEFORE | `["Evaluation of Retrieval-Augmented Generation.pdf"]` — cited a paper for a "don't know" |
| AFTER | `[]` — clean |

Same input, only the prompt changed.

**How do I check whether specific claims in an answer are real?**

A grounding check. A false-premise question (*"Why does the RAG paper conclude
retrieving >10 passages always hurts accuracy?"*) produced very specific claims
(`Figure 3`, `Rouge-L`, `Bleu-1`). A throwaway script printed the retrieved chunks
and searched them for each claim — **all were present**, and the model had actually
*refuted* the false premise using the real text. So the check works both ways: it
flags fabrications *and* clears false alarms. "Does every claim appear in the
retrieved context?" is the natural next gate beyond schema validation.

**Overall finding:** `nvidia/nemotron-3-ultra-550b-a55b:free` is strongly grounded —
it resisted off-topic, specific-fact, and false-premise bait. The one real failure
was the citation-on-refusal, caught by inspection and fixed with a single prompt
rule, verified on identical input.

---

## 7. Rebuilding the same pipeline in LangChain

**I built the gates by hand — what does LangChain replace?**

`PydanticOutputParser(pydantic_object=StructuredAnswer)` does both manual gates in
one object: it parses the JSON *and* validates it against the schema.

**What happens when the model returns malformed JSON — do I still have to skip the question?**

Not necessarily. `OutputFixingParser.from_llm(parser=parser, llm=model)` wraps the
parser and, on a parse failure, sends the broken output **back to the LLM** to be
corrected instead of discarding it. That's a real behavioural difference from the
manual pipeline, which could only log and skip.

**How do I stop hardcoding the schema shape into my prompt text?**

`parser.get_format_instructions()` generates the format description from the
Pydantic model itself, injected via `.partial(format_instructions=...)` — so the
prompt and the schema can't drift apart.

**How do I chain the pieces together?**

The LCEL pipe operator: `prompt | model`, then `conversational | fixing_parser`.
Each stage's output feeds the next.

**How do I give it conversation memory?**

`RunnableWithMessageHistory` wrapping the chain, with `MessagesPlaceholder("history")`
in the prompt template and `InMemoryChatMessageHistory` as the store, keyed by
`session_id` passed at invoke time:

```python
config = {"configurable": {"session_id": "demo1"}}
```

**Does it actually work?** Tested with a deliberately context-dependent follow-up:
Q1 *"What is retrieval-augmented generation?"*, then Q2 *"Who introduced it?"* —
"it" is only resolvable from history.

---

## 8. Comparing two embedding models

**Same PDFs, same chunks, same questions — only the embedding model changes.**
Model 1: `all-MiniLM-L6-v2` (384-dim). Model 2: `all-mpnet-base-v2` (768-dim).
In one line: `chunk once → embed twice → ask the same questions → score each model`.

| Prompt | Answer |
|---|---|
| How do I measure "retrieval quality" objectively? | A **ground-truth hit-rate**: write test questions whose answers I know, then check whether each model retrieves the chunk containing that answer. |
| Should I match on page number or on the answer text? | Answer phrase. Page matching is brittle — my page guesses were wrong 4 of 8 times, answers span pages, and chunk boundaries have nothing to do with the model. |
| What counts as a hit? | `any(phrase in hit["text"] for hit in hits)` — did **at least one** of the top-k chunks contain the answer phrase? Yes = 1, No = 0. |
| Why test at several k values instead of just k=3? | At k=3 both models tied 2/8 — uninformative, since the answer chunk is one needle among 209. Testing k = 3, 5, 10 gives them room to **diverge**. |
| Can't I just compare the distance numbers between the two models? | No. They live in different vector spaces (384-dim vs 768-dim), so distances aren't on the same scale. Distance ranks results *within* a model; hit-rate is what's comparable *across* models. |
| How do I keep the comparison fair? | Change exactly one thing. Same chunks, same questions, same k — and two separate Chroma collections so the embeddings don't overwrite each other. |

**Test questions** — 8 across the 3 papers, each with a distinctive answer phrase:

| # | About | Answer phrase |
|---|---|---|
| 1 | SBERT default pooling | `MEANpooling` |
| 2 | Auepora's three modules | `finalmodule,Metrics` |
| 3 | BART-large parameter count | `406M` |
| 4 | What "EO" stands for | `Evaluable Outputs` |
| 5 | SBERT vs BERT speed | `5 seconds` |
| 6 | Updating RAG's knowledge | `hot-swapping` |
| 7 | Jeopardy factuality vote | `42.7` |
| 8 | SBERT classification objective | `softmax` |

**Result:**

```
   k |   Model 1 (MiniLM) |    Model 2 (mpnet)
------------------------------------------------
   3 |                2/8 |                2/8
   5 |                3/8 |                4/8
  10 |                5/8 |                4/8
```

**Essentially a tie** — 10 hits each across all three k values. mpnet ranks answer
chunks slightly higher at small k; MiniLM catches up and passes at k=10. With only
8 questions a 1-hit difference is within noise.

**Conclusion — the smaller model wins on practicality.** When two models retrieve
about equally well, the lighter one is the better choice: MiniLM is 384-dim vs
mpnet's 768-dim, so it's smaller, faster, and uses less memory for the same
quality. No reason to pay for the heavier model here.

*(The phrase match is a proxy — it assumes "chunk contains the exact answer phrase =
the right chunk." It won't reward a chunk that answers in different words, but both
models are judged by the same rule, so the comparison stays fair.)*

---

## 9. Operational notes

| Prompt | Answer |
|---|---|
| How do I run these? | Always the venv Python: `.venv\Scripts\python.exe <file>.py` — global Python has none of the packages. Ingest once (`python ingest.py`), then query per question. Comparison script: `..\RAG_demo\.venv\Scripts\python.exe compare_EmbeddingModels.py`. |
| Why do the v1 Pydantic methods keep warning? | This is pydantic v2 (`2.13.4`) — use `model_validate`, `model_dump_json`, `model_json_schema`; `.parse_obj` / `.json` / `.schema` are deprecated. |
| How do I get pytest into a venv with no pip? | `.venv\Scripts\python.exe -m ensurepip --upgrade`, then `... -m pip install pytest`. |
| The free model sometimes returns a 502 with `response.choices` as `None` — how do I stop it killing the batch? | `Gate 0`: `generate_answer` raises a clear `RuntimeError`, and the loop catches it and `continue`s to the next question instead of crashing. |
| Why do relative paths break sometimes? | `chroma_db/` resolves against the current working directory — run from the `RAG_demo` folder. |

**Failure gates, end to end:**

```
Gate 0: LLM call fails (502 / rate limit)  → skip question, continue
Gate 1: response isn't valid JSON          → skip question, continue
Gate 2: JSON doesn't match schema          → skip question, continue
        both pass                          → save to outputs.jsonl
```

**Files:**

| File | What it is |
|---|---|
| `ingest.py` | extract → chunk → embed → store (run once) |
| `query.py` | retrieve → build context → generate answer |
| `structured_output.py` | schema + JSON prompt + retrieval + LLM call + gates + save |
| `langchain_pipeline.py` | the same pipeline rebuilt with LCEL, `PydanticOutputParser`, `OutputFixingParser`, and conversation memory |
| `test_validation.py` | 8 pytest tests proving the validator rejects bad output |
| `outputs.jsonl` | validated answers, one JSON object per line (append-only) |
| `.env` | `OPENROUTER_API_KEY` — never commit |
