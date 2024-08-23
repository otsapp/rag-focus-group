
class Config():
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTORSTORE_PATH = "./.chroma_vectorstore"
    VECTOR_SPACE = "cosine"
    NORMALIZE_EMBEDDINGS = True
    DEVICE = "cpu"
    CHUNK_SIZE = 200
    CHUNK_OVERLAP = 20
    LLM = "llama2"
    K_RESULTS = 10