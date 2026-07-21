import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "openai/gpt-oss-20b:free"

messages = [
    {"role": "system", 
     "content": "You are a helpful assistant."}
]

print("Chat started! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break
    else:
        messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    reply = response.choices[0].message.content
    print(f"AI: {reply}\n")

    messages.append({"role": "assistant", "content": reply})
