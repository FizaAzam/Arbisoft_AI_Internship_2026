from typing import Literal
from pydantic import BaseModel, ConfigDict

from dotenv import load_dotenv
from query import generate_answer, retrieve, get_collection

import json
from pydantic import ValidationError

load_dotenv()


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str                                  
    source_files: list[str]                      
    confidence: Literal["high", "medium", "low"]  


SystemPrompt = (
    "You answer questions about a set of research papers using ONLY the provided context.\n"
    "Respond with a SINGLE valid JSON object and NOTHING else - no explanation, no markdown, "
    "no code fences.\n"
    "The JSON object must have exactly these three fields:\n"
    '  "answer": a string containing your answer,\n'
    '  "source_files": a list of the source filenames (strings) you used from the context,\n'
    '  "confidence": one of exactly "high", "medium", or "low".\n'
    "If the answer is not in the context, set answer to say you don't know and confidence to \"low\"."
    "The source_files list must contain ONLY files you actually used to answer. If you don't know or the answer is not in the context, source_files MUST be an empty list [] and confidence MUST be low.\n"

)

if __name__ == "__main__":
    k = 3

    questions = [
        "What exact Exact Match score did RAG achieve on the Natural Questions benchmark?",
        "How many passages are in the Wikipedia index used by the RAG retriever?",
        "Explain why the SBERT paper recommends max-pooling over mean-pooling for the best sentence embeddings.",
        "Why does the RAG paper conclude that retrieving more than 10 passages always hurts accuracy?",
        "Summarize the experiment where the RAG paper combined SBERT embeddings with its DPR retriever." 


    ]

    collection = get_collection("all-MiniLM-L6-v2", "rag_chunks")

    for question in questions:
        print("\n===", question, "===")
        hits = retrieve(collection, question, k)

        try:
            answer = generate_answer(question, hits, SystemPrompt)
        except RuntimeError as e:
            print("LLM call failed - skipping this question:", e)
            continue

        print("Raw Answer from the LLM:\n", answer)

        # GATE 1
        try:
            data = json.loads(answer)     #string to dict
        except json.JSONDecodeError as e:
            print("Gate 1 failed - not valid JSON:", e)
            continue     

        # GATE 2
        try:
            result = StructuredAnswer.model_validate(data)    #dict to verified object
            print("Both gates passed! Validated object:")
            with open("outputs.jsonl", "a", encoding="utf-8") as f:
                f.write(result.model_dump_json() + "\n") #appending as a jsonstring
            print(result)

        except ValidationError as e:
            print("Gate 2 failed - JSON but wrong shape:", e)