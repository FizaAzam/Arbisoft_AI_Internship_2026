import os
from dotenv import load_dotenv

import requests

from langchain.tools import tool

import pdfplumber

load_dotenv()
SERPAPI_API_KEY= os.getenv("SERPAPI_API_KEY")

@tool(parse_docstring=True)
def search_web(query: str) -> str:
    """Search the live web and return the top results with titles, URLs, and snippets.

    Use this whenever the answer depends on current information: recent events,
    news, prices, statistics, or anything that may have changed after your
    training data was collected. Prefer searching over guessing.

    Args:
        query: A focused search query, phrased the way you would type it into a
            search engine. Keep it short and specific rather than pasting the
            user's entire question.
    """

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": 5, 
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=15)
        response.raise_for_status() 
        data = response.json()
    except requests.exceptions.RequestException as exc:
        return f"ERROR: search request failed: {exc}"

    if "error" in data:
        return f"ERROR from SerpApi: {data['error']}"

    results = data.get("organic_results",[])[:5] 
                                    
    if not results:
        return f"No results found for '{query}'."


    formatted = [] 
    for i, item in enumerate(results, start=1):
        title = item.get("title", "Untitled") 
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        formatted.append(f"[{i}] {title}\n    {link}\n    {snippet}")

    return "\n\n".join(formatted)


@tool(parse_docstring=True)
def read_file(path:str)->str:
    """Read the contents of a local .txt or .pdf file and return its text. 
    
    Use this whenever the user refers to a specific local file they want you
    to read, summarize, or answer questions about.

    Args:
        path: The path to the .txt or .pdf file to read, e.g. "notes.txt".
"""

    if not os.path.exists(path):
        return f"Error: {path} not found."
    
    extension = os.path.splitext(path)[1].lower()

    if extension == ".pdf":
        with pdfplumber.open(path) as pdf:
            text="\n".join(page.extract_text() or "" for page in pdf.pages)
        return text
        

    elif extension == ".txt":
        with open(path, "r") as f:
            return f.read()
        

    else:
        return f"Error: {extension}  file type is not supported"
