import pysqlite3
import sys

# fixes an issue with a old version of sqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from config import Config

cfg = Config()


def query(input: str):
    """
    Takes a string as a query and invokes a response from the question, answer chain.

    1. builds the pipeline that defines how the model should be queried
    2. returns the model response

    Parameters
    ----------
    input : str, user query string

    Returns
    -------
    response : **, the response from the large language model 
    """
    pipeline = build_rag_pipeline()
    return pipeline.invoke(input)


def build_rag_pipeline():
    """
    Constructs the rag chain pipeline.

    1. loads embedding model used in constructing the vector store
    2. loads the retriever used to query the chroma vector store
    3. loads the prompt template with question context
    4. loads the chosen large language model 
    5. defines the question answer chain using the loaded components
    
    Returns
    -------
    chain : **, runnable Langchain chain definition
    """

    embedding = get_embeddings(cfg.EMBEDDING_MODEL, cfg.DEVICE, cfg.NORMALIZE_EMBEDDINGS)

    retriever = get_retriever(
        embedding, 
        cfg.VECTORSTORE_PATH, 
        cfg.VECTOR_SPACE,
        cfg.K_RESULTS
        )
    
    prompt = get_prompt_template()
    llm = Ollama(model=cfg.LLM, verbose=False, temperature=0)

    return get_qa_chain(retriever, llm, prompt)


def get_embeddings(embedding_model, device, normalize_embeddings):
     """

    Loads the embedding model.

    Parameters
    ----------
    embedding_model : str, model to be used for generating embeddings for the vector store
    device : str, where embeddings should be processed   
    normalize_embeddings : bool, choice whether to normalise embeddings

    Returns
    -------
    embedding : HuggingFaceEmbeddings, the defined embedding model
    """
     return HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': normalize_embeddings}
    )


def get_retriever(
        embedding: HuggingFaceEmbeddings,
        vectorstore_path: str,
        vector_space: str,
        k_results: int
        ):
    """
    Instatiates the vector database retriever.

    Parameters
    ----------
    embedding : HuggingFaceEmbeddings, embedding model used for vector store
    vectorstore_path : str, where the vector db will be persisted
    vector_space : str, distance measure
    k_result : int, number of results that retriever should return per query
    
    Returns
    -------
    Retriever : **, retriever instance which will query the vector store
    """
    vectorstore = Chroma(persist_directory=vectorstore_path,
                          collection_metadata={"hnsw:space": vector_space},
                          embedding_function=embedding
                          )
    
    return vectorstore.as_retriever(search_kwargs={"k": k_results})


def get_prompt_template():
    """
    Instantiates a langchain prompt template for use with an llm.

    Returns
    -------
    prompt : PromptTemplate, the prompt template instance using a custom template string
    """

    template = """
    The following context is your opinion on cookies: {context}
    Question: {question}

    Do not invent an opinion or answer based on information outside the given context.
    """

    return PromptTemplate.from_template(template)


def format_docs(docs):
    """
    Extracts the text content retrieved from the vector store and returns as a list.

    Parameters
    ----------
    docs : **, documents retrieved from the vector store
    
    Returns
    -------
    text_list : list, list of text content retrieved from the vector store
    """
    return "\n\n".join(doc.page_content for doc in docs)


def inspect(state):
    """
    Prints the state at on point in a Langchain Runnable and passes it on to the next step.

    Parameters
    ----------
    state : **, Langchain runnable
    
    Returns
    -------
    state : **, Langchain runnable
    """
    print(state)
    return state


def get_qa_chain(retriever, llm, prompt):
    """
    Constructs the rag question, answer chain that gets the llm response.

    Parameters
    ----------
    retriever : **, retriever instance which will query the vector store
    llm : **, the chosen large language model
    prompt : PromptTemplate, the prompt template instance using a custom template string 
    
    Returns
    -------
    chain : **, runnable Langchain chain definition
    """
    return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RunnableLambda(inspect)
            | prompt
            | llm
            | StrOutputParser()
        )
