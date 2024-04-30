import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
# from langchain_openai import OpenAIEmbeddings
from langchain.globals import set_verbose

load_dotenv()
set_verbose(True)

embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2", show_progress=True)
# embedding_function = OpenAIEmbeddings(
#     model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"), chunk_size=1000, show_progress_bar=True)


def init_db():
    loader = CSVLoader(file_path='./data/output.csv',
                       source_column="Vehicle_Title")
    data = loader.load()
    print("Embedding")
    # db = Chroma.from_documents(data, embedding_function)
    # db.save_local("reviews")
    db = Chroma.from_documents(
        data, embedding_function, persist_directory="./chroma_db")
    print("Done embedding")
    return db

def init_listings():
    loader = CSVLoader(file_path='./data/list.csv',
                       source_column="URL")
    data = loader.load()
    print("Embedding")
    # db = Chroma.from_documents(data, embedding_function)
    # db.save_local("reviews")
    db = Chroma.from_documents(
        data, embedding_function, persist_directory="./cars_db")
    print("Done embedding")
    return db


def load_db():
    # db = FAISS.load_local("reviews", embedding_function,
    #                       allow_dangerous_deserialization=True)
    db = Chroma(persist_directory="./chroma_db",
                embedding_function=embedding_function)
    retriever = db.as_retriever()

    return retriever

def load_listings():
    # db = FAISS.load_local("reviews", embedding_function,
    #                       allow_dangerous_deserialization=True)
    db = Chroma(persist_directory="./cars_db",
                embedding_function=embedding_function)
    retriever = db.as_retriever()

    return retriever


if __name__ == "__main__":
    init_listings()
