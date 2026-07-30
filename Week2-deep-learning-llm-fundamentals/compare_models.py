import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

models = [
    "openai/gpt-oss-20b:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

prompt = "Explain what an API is to a 12-year-old in 3 sentences."

for model in models:
    print("\n")
    print(f"MODEL: {model}")
    print("\n")

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.time() - start
        print(response.choices[0].message.content)
        print(f"\n[Time taken: {elapsed:.2f} seconds]")
    except Exception as e:
        print(f"[This model is unavailable right now: {e}]")

