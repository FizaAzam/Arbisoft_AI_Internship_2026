import os

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_collection(ModelName, CollectionName):
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name= ModelName
    )
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_collection(name= CollectionName, embedding_function=embed_fn)


def retrieve(collection, question, k):
    results = collection.query(query_texts=[question], n_results=k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    hits = []
    for text, meta, dist in zip(documents, metadatas, distances):
        hits.append({"text": text, "meta": meta, "distance": dist})
    return hits


def build_context(hits):
    parts = []
    for hit in hits:
        meta = hit["meta"]
        parts.append(f"[{meta['source']}, page {meta['page']}]\n{hit['text']}")
    return "\n\n".join(parts)


SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a set of research "
    "papers. Use ONLY the context provided to answer. If the answer is not in "
    "the context, say you don't know. Cite the source file and page number for "
    "any facts you use."
)

def generate_answer(question, hits, SYSTEM_PROMPT):
    context = build_context(hits)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    if response.choices is None:
        raise RuntimeError(f"LLM call failed: {getattr(response, 'error', 'unknown error')}")
    return response.choices[0].message.content


