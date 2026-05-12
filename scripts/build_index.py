import os
import pandas as pd
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from loguru import logger

load_dotenv()

DATA_PATH = "data/raw/mtsamples.csv"
INDEX_PATH = "data/faiss_index"

logger.add("build_index.log", rotation="10 MB")


def load_dataset():

    logger.info("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["transcription"])

    logger.info(f"Loaded {len(df)} valid rows")

    return df


def create_documents(df):

    logger.info("Creating text chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    docs = []

    for _, row in df.iterrows():

        transcription = str(row["transcription"])

        specialty = str(
            row.get("medical_specialty", "Unknown")
        )

        chunks = splitter.split_text(transcription)

        for chunk in chunks:

            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "specialty": specialty
                    }
                )
            )

    logger.info(f"Created {len(docs)} chunks")

    return docs


def load_embeddings():

    logger.info("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


def build_index(documents, embeddings):

    logger.info("Building FAISS index...")

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    logger.info("Saving FAISS index...")

    vectorstore.save_local(INDEX_PATH)

    logger.info("FAISS index saved successfully")


def main():

    logger.info("Starting build process")

    df = load_dataset()

    docs = create_documents(df)

    embeddings = load_embeddings()

    build_index(docs, embeddings)

    logger.info("Build completed successfully")


if __name__ == "__main__":
    main()
