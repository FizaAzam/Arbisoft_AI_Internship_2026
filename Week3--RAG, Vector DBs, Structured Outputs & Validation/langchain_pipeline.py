import os
from langchain_openai import ChatOpenAI                      
from langchain_core.prompts import ChatPromptTemplate  

from langchain_core.output_parsers import PydanticOutputParser  
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.prompts import MessagesPlaceholder 
from langchain_core.runnables.history import RunnableWithMessageHistory  
from langchain_core.chat_history import InMemoryChatMessageHistory      

from structured_output import StructuredAnswer

from query import get_collection, retrieve, build_context


model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],  
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
)

parser = PydanticOutputParser(pydantic_object=StructuredAnswer) #json.loads() + validation
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You answer questions about a set of research papers using ONLY the "
         "provided context. If the answer is not in the context, say you don't "
         "know and set confidence to \"low\".\n{format_instructions}"),
         MessagesPlaceholder("history"),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational = RunnableWithMessageHistory(
    prompt | model,                 
    get_session_history,
    input_messages_key="question",  
    history_messages_key="history", 
)

chain = conversational | fixing_parser   


if __name__ == "__main__":
    collection = get_collection("all-MiniLM-L6-v2", "rag_chunks")

    config = {"configurable": {"session_id": "demo1"}}


    q1 = "What is retrieval-augmented generation?"
    context1 = build_context(retrieve(collection, q1, k=3))
    result1 = chain.invoke({"context": context1, "question": q1}, config=config)
    print("Q1:", q1)
    print(result1, "\n")


    q2 = "Who introduced it?"
    context2 = build_context(retrieve(collection, q2, k=3))
    result2 = chain.invoke({"context": context2, "question": q2}, config=config)
    print("Q2:", q2)
    print(result2)


