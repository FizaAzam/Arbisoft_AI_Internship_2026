import sys
import pathlib
 
RAG_DEMO_DIR = pathlib.Path(__file__).resolve().parent.parent / "RAG_demo"
sys.path.insert(0, str(RAG_DEMO_DIR))

PDF_DIR = RAG_DEMO_DIR / "pdfs"

from ingest import extract_pages, chunk_pages, embed_and_store
from query import retrieve

pages = extract_pages(PDF_DIR)
chunks = chunk_pages(pages)

FirstModel = "all-MiniLM-L6-v2"
FirstCollectionName = "Firstcollection"
FirstCollection = embed_and_store(chunks, FirstModel, FirstCollectionName)

SecondModel = "all-mpnet-base-v2"
SecondCollectionName = "Secondcollection"
SecondCollection = embed_and_store(chunks, SecondModel, SecondCollectionName)


# 8 test questions, each with a distinctive answer phrase that was verified to actually exist in the extracted chunks.
TEST_QUERIES = [
    {"question": "What pooling strategy does SBERT use by default?",
     "answer_contains": "MEANpooling"},
    {"question": "What are the three modules of the Auepora evaluation framework?",
     "answer_contains": "finalmodule,Metrics"},
    {"question": "How many trainable parameters does the BART-large generator have in the original RAG paper?",
     "answer_contains": "406M"},
    {"question": "What does 'EO' stand for in the Auepora framework?",
     "answer_contains": "Evaluable Outputs"},
    {"question": "How much faster is SBERT than BERT at finding the most similar sentence pair in a set of 10,000 sentences?",
     "answer_contains": "5 seconds"},
    {"question": "Why can a RAG model update its knowledge without needing to be retrained?",
     "answer_contains": "hot-swapping"},
    {"question": "What percentage of the time did human evaluators judge RAG's generations as more factual than BART's in the Jeopardy question generation task?",
     "answer_contains": "42.7"},
    {"question": "What objective function does SBERT use when fine-tuning on labeled classification data like SNLI?",
     "answer_contains": "softmax"},
]



K_VALUES = [3, 5, 10]
n = len(TEST_QUERIES)

print("\n")
print("RETRIEVAL QUALITY COMPARISON  (hit = answer phrase found in top-k chunks)")
print(f"Model 1 = {FirstModel}")
print(f"Model 2 = {SecondModel}")
print("\n")

summary = {}  

for k in K_VALUES:
    model1_hits = 0
    model2_hits = 0

    print(f"\n--- k = {k} ---")
    for q in TEST_QUERIES:
        question = q["question"]
        phrase = q["answer_contains"]

        hits1 = retrieve(FirstCollection, question, k)
        hits2 = retrieve(SecondCollection, question, k)

        found1 = any(phrase.lower() in hit["text"].lower() for hit in hits1)
        found2 = any(phrase.lower() in hit["text"].lower() for hit in hits2)

        model1_hits += found1   
        model2_hits += found2

        mark1 = "OK  " if found1 else "MISS"
        mark2 = "OK  " if found2 else "MISS"
        print(f"  M1 {mark1} | M2 {mark2} | {question[:52]}")

    summary[k] = (model1_hits, model2_hits)
    print(f"  -> Recall@{k}:  Model 1 = {model1_hits}/{n}   Model 2 = {model2_hits}/{n}")



print("\n")
print("========SUMMARY (Recall@k)==========")
print("\n")
print(f"{'k':>4} | {'Model 1 (MiniLM)':>18} | {'Model 2 (mpnet)':>18}")

for k in K_VALUES:
    m1, m2 = summary[k]
    print(f"{k:>4} | {f'{m1}/{n}':>18} | {f'{m2}/{n}':>18}")

