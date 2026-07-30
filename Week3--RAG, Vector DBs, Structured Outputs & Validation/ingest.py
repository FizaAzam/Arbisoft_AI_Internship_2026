import pathlib
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

PDF_DIR = pathlib.Path("pdfs")


def extract_pages(pdf_dir):
    records = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()

                if not text:
                    continue

                records.append(
                    {
                        "source": pdf_path.name,
                        "page": page.page_number,
                        "text": text,
                    }
                )

    return records


def chunk_pages(records):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = []
    for record in records:
        for piece in splitter.split_text(record["text"]):
            chunks.append(
                {
                    "source": record["source"],
                    "page": record["page"],
                    "text": piece,
                }
            )
    return chunks


def embed_and_store(chunks, ModelName, CollectionName):
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name = ModelName
    )

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(
        name= CollectionName,
        embedding_function=embed_fn,
    )

    ids = []
    documents = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        ids.append(f"chunk: {i}")
        documents.append(chunk["text"])
        metadatas.append({"source": chunk["source"], "page": chunk["page"]})

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return collection

