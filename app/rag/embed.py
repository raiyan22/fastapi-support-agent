from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PERSIST_DIR = str(Path(__file__).resolve().parent.parent.parent / "chroma_db")
COLLECTION_NAME = "support_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def load_single_file(file_path: str):
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return loader.load()


def embed_file(file_path: str, doc_id: str) -> int:
    docs = load_single_file(file_path)

    for doc in docs:
        doc.metadata["doc_id"] = doc_id

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id

    embeddings = get_embedding_function()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )

    return len(chunks)


def delete_from_chroma(doc_id: str):
    embeddings = get_embedding_function()
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    vectorstore.delete(where={"doc_id": doc_id})
