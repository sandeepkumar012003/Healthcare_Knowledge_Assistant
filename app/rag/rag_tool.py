import os

from functools import lru_cache

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    "data/faiss_index"
)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def rag_query(query: str, k: int = 10):

    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search_with_score(
        query,
        k=k
    )

    output = []
    seen = set()

    for doc, score in results:

        chunk = doc.page_content.strip()

        # Remove duplicate chunks
        if chunk in seen:
            continue

        seen.add(chunk)

        # Keep only better matches
        if score < 1.2:

            output.append(
                {
                    "chunk": chunk,
                    "score": float(score)
                }
            )

    return output
