from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PERSIST_DIR = "../chroma_db"
COLLECTION_NAME = "support_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

def load_documents(data_dir: str = "data"):
    data_path = Path(data_dir)
    documents = []
    
    print(f"Looking for files in: {data_path.absolute()}")
    
    # Load txt files
    for txt_file in data_path.rglob("*.txt"):
        try:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            documents.extend(loader.load())
            print(f"Loaded: {txt_file.name}")
        except Exception as e:
            print(f"Error loading {txt_file.name}: {e}")

    return documents

def ingest_documents(force_reingest: bool = True):
    documents = load_documents()
    if not documents:
        print("No documents found.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = get_embedding_function()
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )
    
    print(f"Successfully ingested {len(chunks)} chunks!")

if __name__ == "__main__":
    ingest_documents()