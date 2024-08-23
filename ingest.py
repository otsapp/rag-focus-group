import os
import shutil
import pysqlite3
import sys

# fixes an issue with a old version of sqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import Config

cfg = Config()


def run_ingest():
    docs = load_docs('texts/cookie_reviews.csv')
    chunks = chunk_docs(docs, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    build_vector_store(chunks, 
                       cfg.EMBEDDING_MODEL, 
                       cfg.NORMALIZE_EMBEDDINGS, 
                       cfg.DEVICE,
                       cfg.VECTORSTORE_PATH,
                       cfg.VECTOR_SPACE)


def load_docs(path: str):
    """

    Constructs and persists a Chroma db vector store for huggingface 
    embeddings.

    Parameters
    ----------
    path : str, path to text documents.

    Returns
    -------
    docs : **, loaded text documents.

    """
    loader = CSVLoader(file_path=path)
    return loader.load()


def chunk_docs(docs, chunk_size: int, chunk_overlap: int):
    """

    Constructs and persists a Chroma db vector store for huggingface 
    embeddings.

    Parameters
    ----------
    docs : **, text documents.

    Returns
    -------
    chunks : **, chunks split from the raw text documents.

    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return text_splitter.split_documents(docs)


def build_vector_store(
        chunks, 
        embedding_model: str, 
        normalize_embeddings: bool, 
        device: str, 
        vectorstore_path: str, 
        vector_space: str
        ) -> None:
    """

    Constructs and persists a Chroma db vector store for huggingface 
    embeddings.

    1. Defines model used to generate embeddings.
    2. Builds vector store with embeddings generated from text chunks.

    Parameters
    ----------
    chunks : **, chunks split from the raw text documents.
    embedding_model : str, model to be used for generating embeddings for the vector store
    normalize_embeddings : bool, choice whether to normalise embeddings
    device : str, where embeddings should be processed
    vectorstore_path : str, where the vector db will be persisted
    vector_space : str, distance measure. 
    collection_name : str, name of vector store collection
    
    """

    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': normalize_embeddings}
    )

    if os.path.isdir(vectorstore_path):
        shutil.rmtree(vectorstore_path)

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=vectorstore_path,
        collection_metadata={"hnsw:space": vector_space}
     )
    