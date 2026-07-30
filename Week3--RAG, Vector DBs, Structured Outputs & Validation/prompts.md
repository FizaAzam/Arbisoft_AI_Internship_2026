# RAG Demo — Chat With Your PDFs (Week 3)

A small Retrieval-Augmented Generation (RAG) system that answers questions about a set of PDFs.
Text is extracted, chunked, embedded **locally**, and stored in **ChromaDB**; at query time the
most relevant chunks are retrieved and passed to an LLM (via OpenRouter) to produce a **grounded,
cited answer**.

---

## Concepts I Learned (the questions that unlocked each idea)

These are the questions I worked through while building this demo - each one taught a core concept:

1. **What is RAG, conceptually?** — Retrieve relevant passages from *my* documents, then let the
   LLM answer *using those passages* → grounded answers, less hallucination.
2. **Why do we need chunking?** — Whole pages make blurry embeddings; small focused chunks make
   sharp ones, so retrieval matches the question precisely (and we feed the LLM only what's
   relevant).
3. **What is semantic chunking (vs. what I used)?** — I used recursive fixed-size splitting;
   semantic chunking cuts where the *meaning* shifts instead of at character counts.
4. **What is a local embedding generator? Is it the sentence-transformer?** — Yes: the
   `all-MiniLM-L6-v2` model, run via sentence-transformers, turns text into vectors on my own
   machine (free, offline, private).
5. **What does "local" mean?** — Runs on my computer: free, offline after one download, private
   (text never leaves the machine), but bounded by my hardware and a smaller model.
6. **What is a vector database / ChromaDB?** — Stores embeddings (+ text + metadata) and lets me
   search by *meaning* (nearest vectors), not exact keywords.
7. **What is a collection?** — A table inside the vector DB; each row = one chunk's id + text +
   embedding + metadata. `collection.count()` = number of stored chunks.
8. **What is `PersistentClient()`?** — Opens/creates a ChromaDB that lives on disk, so I embed
   once (ingest) and reuse the vectors on every query.
9. **What does `collection.query()` do?** — Embeds the question with the *same* model, compares it
   to every stored vector, and returns the closest `k` chunks, ranked by distance.
10. **What is in `results`, and why `["field"][0]`?** — A dict of parallel list-of-lists; the outer
    list has one slot per question, so `[0]` unwraps my single question's matches.
11. **What does `zip()` do?** — Walks the parallel `documents`/`metadatas`/`distances` lists
    together so each chunk's info gets bundled into one `hit`.
12. **What is `build_context()` for?** — Flattens the retrieved chunks into one labeled text block
    (`[source, page]` + text) that becomes the LLM's reference material.
13. **What is the system prompt?** — The standing instruction that tells the LLM to answer ONLY from
    the provided context and cite sources — the guardrail that keeps answers grounded.
14. **Why sort in `glob("*.pdf")`?** — Guarantees PDFs are read in the same order every run, so
    chunk IDs stay stable and re-ingesting overwrites cleanly.
15. **Why not just paste a PDF into ChatGPT/Claude?** — Works for one small doc, but RAG scales to
    many/large documents, costs less (only relevant chunks are sent), and persists as a reusable
    knowledge base.

---

## Pipeline

```
ingest.py  (run once)                  query.py  (run per question)
─────────────────────                  ────────────────────────────
1. Extract text   (pdfplumber)         5. Retrieve      (Chroma similarity search)
2. Chunk          (LangChain splitter) 6. Build context (+ [source, page] labels)
3. Embed          (MiniLM, local)      7. Generate      (LLM via OpenRouter)
4. Store          (ChromaDB on disk)
```

## Tech Stack

| Stage | Tool |
|-------|------|
| PDF text extraction | `pdfplumber` |
| Chunking | `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) |
| Embeddings (local) | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector database | `chromadb` (`PersistentClient`) |
| Answer generation | `openai` library pointed at OpenRouter (free model) |
| Secrets | `python-dotenv` (`.env`) |

## Project Structure

```
RAG_demo/
├── pdfs/            # source PDFs
├── chroma_db/       # persisted vector database (created by ingest.py)
├── ingest.py        # steps 1-4: extract -> chunk -> embed -> store  (run once)
├── query.py         # steps 5-6: retrieve -> generate answer         (run per question)
├── .env             # OPENROUTER_API_KEY (never commit this)
└── README.md
```

## Setup

```bash
uv venv
uv pip install pdfplumber langchain-text-splitters chromadb sentence-transformers openai python-dotenv
```

Create a `.env` file with a free OpenRouter API key:

```
OPENROUTER_API_KEY=sk-or-your-key-here
```

## Usage

**1. Build the vector database (run once, or whenever the PDFs change):**

```bash
python ingest.py
```
```
Extracted 51 pages from pdfs/
Split into 209 chunks
Stored 209 chunks in ChromaDB (chroma_db/)
```

**2. Ask a question:**

```bash
python query.py
```
```
Question: What is retrieval-augmented generation?

Retrieved chunks:
  1. Evaluation of Retrieval-Augmented Generation.pdf, page 18 (distance 0.326)
  2. Evaluation of Retrieval-Augmented Generation.pdf, page 1  (distance 0.368)
  3. Evaluation of Retrieval-Augmented Generation.pdf, page 19 (distance 0.374)

Answer:
Retrieval-augmented generation (RAG) is ... [Evaluation of Retrieval-Augmented Generation.pdf, page 1].
```

(Edit the `question` variable in `query.py` to ask something else.)

## Configuration Knobs

| Setting | File | Meaning |
|---------|------|---------|
| `CHUNK_SIZE` | `ingest.py` | max characters per chunk (default 1000) |
| `CHUNK_OVERLAP` | `ingest.py` | shared characters between neighbouring chunks (default 150) |
| `TOP_K` | `query.py` | how many chunks to retrieve per question (default 3) |
| `LLM_MODEL` | `query.py` | the OpenRouter model (use a free `:free` slug) |

## Notes

- **Embeddings are local; only generation uses the internet.** Only the question and retrieved
  chunks are sent to OpenRouter.
- **Free LLM models only** — swap `LLM_MODEL` for any `:free` slug from
  [openrouter.ai/models](https://openrouter.ai/models).
- **`.env` holds a secret** — add it to `.gitignore` before committing.

---

# Structured-Output Pipeline (Week 3)

Extends the RAG demo. Instead of returning a free-text answer, this task makes the
LLM return **validated, structured JSON**, then validates it, saves it, and documents where
the model misbehaved.

**The pipeline in one line:** `LLM → JSON → validate → save`.

The only *new* work versus the RAG demo is the middle two steps (parse + validate); retrieval
and the LLM call are reused from `query.py`.

---

## Files

| File | What it is |
|---|---|
| `structured_output.py` | The pipeline: schema + prompt + retrieval + LLM call + gates + save |
| `query.py` | Reused from Task 1 — `get_collection`, `retrieve`, `generate_answer` |
| `test_validation.py` | 8 pytest tests that prove the validator rejects bad output |
| `outputs.jsonl` | Validated answers, one JSON object per line (append-only) |

**Run the pipeline:**
```
.venv\Scripts\python.exe structured_output.py
```
**Run the tests:**
```
.venv\Scripts\python.exe -m pytest test_validation.py -v
```

---

## How it was built (step by step)

### 1. The schema (the "gate")
A Pydantic v2 model defines what a *valid* answer looks like. The type hints **are** the
validation rules — no manual checking needed.

```python
class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")   # reject invented fields
    answer: str
    source_files: list[str]
    confidence: Literal["high", "medium", "low"]  # only these 3 values
```

Verified by hand in the REPL before building anything on top: a good payload returns an
object; bad enum / wrong type / extra field each raise `ValidationError`.

### 2. The prompt
A new `SystemPrompt` (different from `query.py`'s prose prompt) that demands a **single JSON
object** with exactly those three fields — no markdown, no prose. The schema's shape is spelled
out in the prompt so the model knows what to produce.

### 3. Retrieval + LLM call (reused)
Reuses `get_collection` + `retrieve` from `query.py` to get the context, then
`generate_answer(question, hits, SystemPrompt)`. `generate_answer` was refactored to take the
system prompt as an argument so this task could pass its JSON-demanding prompt while Task 1's
behaviour stayed available.

### 4. The two gates (the heart of the task)
The LLM hands back a **string** that cannot be trusted. Two separate checkpoints turn it into
either a verified object or a logged rejection:

```
Gate 1 — json.loads(raw)                 → catches JSONDecodeError (fences, prose, cut-off text)
Gate 2 — StructuredAnswer.model_validate → catches ValidationError (wrong type / field / enum)
```

Only if **both** pass does anything reach disk. A `Gate 0` was added later to handle the LLM
*call itself* failing (see Setup notes).

### 5. Save
Validated output is appended to `outputs.jsonl` with `result.model_dump_json()`. Invalid output
is never saved — it's logged and skipped.

---

## Validation tests (`test_validation.py`)

The tests do **not** call the LLM. They hand the validator hand-written malformed payloads and
assert each is rejected. A test using `pytest.raises(...)` **passes when the gate correctly
throws out bad data** — so green means the validation works.

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

Result: **8 passed** — and the suite runs without firing the LLM, thanks to the
`if __name__ == "__main__":` guard that keeps importing the module side-effect-free.

---

## Hallucination documentation

Task requirement: *document where the LLM hallucinated and how it was caught.*

### Test setup
Three questions were run: two out-of-context (**capital of France**, **quicksort time
complexity** — nothing about these is in the RAG papers) and one in-context (**RAG**) as a
control.

### Finding 1 — the guardrail prompt prevented fabrication
With the prompt line *"If the answer is not in the context, say you don't know and set
confidence to low"* present, the model **honestly refused** both out-of-context questions:
`answer` said "not in the context", `confidence: "low"`, `source_files: []`. No hallucination.
This is a positive result — the guardrail worked.

### Finding 2 — removing the guardrail produced a fabricated citation
Commenting out that one prompt line and re-running, the **France** question returned:

```json
{
  "answer": "The provided context does not contain information about the capital of France...",
  "source_files": ["Evaluation of Retrieval-Augmented Generation.pdf",
                   "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf"],
  "confidence": "low"
}
```

The answer says *"I don't know — no info about France"*, yet it **cited two source papers**.
Citing sources for a non-answer is self-contradictory — a **fabricated citation**. Those papers
contain nothing about France.

**How it was caught — and why validation alone couldn't catch it:** the JSON is structurally
perfect (a valid list of valid strings), so **both gates passed it**. Schema validation
guarantees *structure, not truth*. Catching this needs a separate semantic check — e.g.
*"is every cited file actually one of the retrieved hits?"* — which Pydantic cannot express.

### Finding 3 — confidence drifted upward without the guardrail
On the in-context RAG question, removing the guardrail nudged `confidence` from `medium` to
`high`. The model got less cautious across the board.

### Comparison

| Question | Guardrail ON | Guardrail OFF |
|---|---|---|
| quicksort | refuses, `sources: []` | refuses, `sources: []` (still clean) |
| France | refuses, `sources: []` | refuses **but cites 2 papers** (fabricated citation) |
| RAG (control) | correct, `confidence: medium` | correct, `confidence: high` |

**Takeaway:** one prompt line was the difference between a clean refusal and a self-contradictory
one. And the fabricated citation shows the boundary of schema validation — it enforces shape,
not correctness.

---

## Citation-on-refusal hallucination, and the prompt fix

Findings 1–3 above used *obviously* out-of-context questions (France, quicksort). To probe the
realistic **gray zone**, a second experiment used **in-domain** questions that ask for a
**specific fact** the papers cover but that isn't in the top-3 retrieved chunks — e.g.
*"How many passages are in the Wikipedia index used by the RAG retriever?"*. The strong guardrail
was kept **on** the whole time (the prompt was never weakened).

### The hallucination (BEFORE the fix)

Even with the guardrail on, the model said it didn't know **yet still cited a source**:

```json
{
  "answer": "The provided context does not mention the number of passages in a Wikipedia index used by a RAG retriever.",
  "source_files": ["Evaluation of Retrieval-Augmented Generation.pdf"],
  "confidence": "low"
}
```

The prompt defines `source_files` as *"the source filenames you **used** from the context."*
Here the model **used nothing** (it said "does not mention"), yet listed a paper anyway — a
**fabricated citation**: claiming support from a document for an answer it admits it doesn't have.
(It was also inconsistent — other "don't know" answers in the same run correctly returned `[]`.)

### How it was caught

It was caught by **reading the output**: the `answer` text
("does not mention…") directly contradicts a non-empty `source_files`. 

### The fix

One line was **added** to the system prompt (strengthening it, not weakening it):

```
The source_files list must contain ONLY files you actually used to answer. If you don't know
or the answer is not in the context, source_files MUST be an empty list [] and confidence MUST be low.
```

### Result (AFTER the fix)

Re-running the **same question** with the new rule:

```json
{
  "answer": "I don't know. The provided context does not mention the number of passages in the Wikipedia index used by the RAG retriever.",
  "source_files": [],
  "confidence": "low"
}
```

Same input, only the prompt changed — the fabricated citation is gone (`source_files: []`).

| | `source_files` |
|---|---|
| BEFORE (no rule) | `["Evaluation of Retrieval-Augmented Generation.pdf"]` — cited a paper for a "don't know" |
| AFTER (rule added) | `[]` — clean |

### Bonus — a grounding check that catches *and* clears

A false-premise question (*"Why does the RAG paper conclude retrieving >10 passages always hurts
accuracy?"*) produced an answer with very specific claims (`Figure 3`, `Rouge-L`, `Bleu-1`). To
check whether those were fabricated, a throwaway script printed the retrieved chunks and searched
them for each claim. **All were present** — the model had grounded its answer in the real
retrieved text (it even *refuted* the false premise using it). So the grounding check works both
ways: it flags fabricated claims *and* clears false alarms. That "does every claim appear in the
retrieved context?" test is the natural next gate beyond schema validation.

**Overall finding:** this model (`nvidia/nemotron-3-ultra-550b-a55b:free`) is strongly grounded —
it resisted off-topic, specific-fact, and false-premise bait. The one real failure was the
citation-on-refusal above, caught by inspection and fixed with a single prompt rule, verified on
identical input.

---

## Setup notes 

- **Always use the venv Python:** `.venv\Scripts\python.exe <file>.py`. Global Python has none
  of the packages.
- **pydantic v2** (`2.13.4`) — use `model_validate`, `model_dump_json`, `model_json_schema`
  (the v1 `.parse_obj` / `.json` / `.schema` are deprecated).
- **pytest** had to be bootstrapped into the venv (no pip initially):
  `.venv\Scripts\python.exe -m ensurepip --upgrade` then `... -m pip install pytest`.
- **`if __name__ == "__main__":` guard** wraps the pipeline run so importing the module (in the
  tests) doesn't fire a real LLM call.
- **Free model only** (`nvidia/nemotron-3-ultra-550b-a55b:free` via OpenRouter). It occasionally
  returns a transient **502 "workers busy"** error where `response.choices` is `None`. `Gate 0`
  handles this: `generate_answer` raises a clear `RuntimeError`, and the loop catches it and
  skips that question (`continue`) instead of crashing the whole batch.
- Relative paths (`chroma_db/`) resolve to the current working dir — run from the `RAG_demo`
  folder.

---

## Failure gates summary

```
Gate 0: LLM call fails (502 / rate limit)  → skip question, continue
Gate 1: response isn't valid JSON          → skip question, continue
Gate 2: JSON doesn't match schema          → skip question, continue
        both pass                          → save to outputs.jsonl
```

---

# Comparing Two Embedding Models on Retrieval Quality

Reuses this RAG demo's ingest pipeline, but instead of answering questions, this task
**compares two local embedding models** to see which one retrieves better chunks.
Lives in the sibling folder `../compare_EmbeddingModels/`.

- **Model 1:** `all-MiniLM-L6-v2` (small, fast, 384-dim)
- **Model 2:** `all-mpnet-base-v2` (larger, higher quality, 768-dim)

Same PDFs, same chunks, same questions — only the embedding model changes. Whichever model
more often retrieves the chunk that actually contains the answer is the better retriever.

**The task in one line:** `chunk once → embed twice → ask the same questions → score each model`.

---

## Concepts I Learned 

1. **How do I measure "retrieval quality"?** — I use a **ground-truth hit-rate**:
   write test questions whose answers I know, then check whether each model retrieves the chunk
   containing that answer.
2. **Why match on an answer phrase instead of a page number?** — Page matching is brittle: my page
   guesses were wrong 4 of 8 times, answers can span pages, and chunk boundaries have nothing to do
   with the model. Instead I check whether a **distinctive answer phrase** appears in the retrieved
   chunk text.
3. **What counts as a "hit"?** — `any(phrase in hit["text"] for hit in hits)`: did **at least one** of
   a model's top-k retrieved chunks contain the answer phrase? Yes = 1, No = 0.
4. **Why test at several k values (Recall@k)?** — At k=3 both models tied 2/8 — uninformative, because
   the answer chunk is one needle among 209. Testing at k = 3, 5, 10 gives each model more room and
   lets them **diverge**, showing which is actually better.
5. **Why not just compare the distance numbers between models?** — The two models live in different
    vector spaces (384-dim vs 768-dim), so their distances aren't on the same scale. Distance ranks
    results *within* one model; hit-rate is what's comparable *across* models.

---

## Pipeline

```
1. Extract text   (pdfplumber)          reused from RAG demo
2. Chunk          (LangChain splitter)  reused - done ONCE
3. Embed twice    (MiniLM -> collection 1,  mpnet -> collection 2)
4. Ask 8 test questions, retrieve top-k from each model
5. Hit = answer phrase found in the retrieved chunks
6. Score = Recall@k for each model, at k = 3, 5, 10
```

## Test questions

8 questions spread across the 3 papers, each with a distinctive answer phrase:

| # | Question is about | Answer phrase |
|---|---|---|
| 1 | SBERT default pooling | `MEANpooling` |
| 2 | Auepora's three modules | `finalmodule,Metrics` |
| 3 | BART-large parameter count | `406M` |
| 4 | What "EO" stands for | `Evaluable Outputs` |
| 5 | SBERT vs BERT speed | `5 seconds` |
| 6 | Updating RAG's knowledge | `hot-swapping` |
| 7 | Jeopardy factuality vote | `42.7` |
| 8 | SBERT classification objective | `softmax` |

## Run it

```
..\RAG_demo\.venv\Scripts\python.exe compare_EmbeddingModels.py
```

(Uses this RAG_demo venv, which already has all the packages. Reads PDFs from `../RAG_demo/pdfs`.)

## Notes

- **Fair comparison rule:** change only one thing (the embedding model). Same chunks, same questions,
  same k — so any difference in the score is the model's doing.
- **Two collections, not one** — each model's embeddings live in a separate ChromaDB collection so
  they don't overwrite each other.
- **The phrase match is a proxy** — it assumes "chunk contains the exact answer phrase = the right
  chunk was retrieved." It won't reward a chunk that answers in different words, but both models are
  judged by the same rule, so the comparison stays fair.

---

## Result

```
   k |   Model 1 (MiniLM) |    Model 2 (mpnet)
------------------------------------------------
   3 |                2/8 |                2/8
   5 |                3/8 |                4/8
  10 |                5/8 |                4/8
```

**It's essentially a tie.** Adding up all three k values: MiniLM = 10 hits, mpnet = 10 hits — dead
even. The small pattern: mpnet is slightly better at small k (top 3–5, it ranks answer chunks a
touch higher), while MiniLM catches up and passes at k=10. With only 8 questions, a 1-hit difference
is within noise, so the honest read is **both models retrieve at roughly the same quality on this
test set.**

**Conclusion — the smaller model wins on practicality.** When two models retrieve about equally
well, the lighter one is the better choice: MiniLM is 384-dim vs mpnet's 768-dim, so it's smaller,
faster, and uses less memory for the same retrieval quality. There's no reason to pay for the
heavier model here.
