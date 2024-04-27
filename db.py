from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2")


def init_db():
    loader = CSVLoader(file_path='./data/output.csv',
                       source_column="Vehicle_Title")
    data = loader.load()
    db = FAISS.from_documents(data, embedding_function)
    db.save_local("reviews")


def load_db():
    db = FAISS.load_local("reviews", embedding_function,
                          allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    return retriever


if __name__ == "__main__":
    init_db()
