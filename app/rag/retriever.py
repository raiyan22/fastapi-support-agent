from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict

CHROMA_PERSIST_DIR = "../chroma_db"
COLLECTION_NAME = "support_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_function():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

def search_knowledge(query: str, k: int = 4) -> List[Dict]:
    embeddings = get_embedding_function()
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    docs = vectorstore.similarity_search(query, k=k)
    
    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content.strip(),
            "metadata": doc.metadata
        })
    return results

if __name__ == "__main__":
    results = search_knowledge("return policy")
    print(f"Found {len(results)} results")
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(r["content"][:400])