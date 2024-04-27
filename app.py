import os
from dotenv import load_dotenv
# from flask_cors import CORS
# from flask import Flask, request
import streamlit as st

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from langchain import hub
from db import load_db


load_dotenv()


text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

SYSTEM_PROMPT = (
    "You are a helpful AI car salesman. A user is looking for a car. "
    "The provided context is a dataset containing car models that is related to the preferences of the user. "
    "The dataset includes the specific car model and the summarized review of the particular car. "
    "Please analyse the cars and its reviews and preferences of the users and return the best 5 cars they should consider. "
    "----------------"
    "{context}"
    "----------------"
)

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{input}")]
)

# app = Flask(__name__)
# CORS(app)


def prompt(query):
    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    print(query)

    retriever = load_db()
    # model = Ollama(base_url='http://119.74.32.2:11434',
    #                model="llama2", temperature=0.8)
    model = ChatOpenAI(model="gpt-3.5-turbo",
                       api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

    # qachain = RetrievalQA.from_chain_type(model, retriever=retriever)

    question_answer_chain = create_stuff_documents_chain(
        model, PROMPT)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    output = chain.invoke({"input": query})
    # output = qachain.invoke({"query": prompt})
    print(output)
    return {"response": output["answer"]}


st.title("📓 Matt's Black Book")
resp = st.text_input("Enter your query here", key="query")
if st.button("Submit"):
    response = prompt(resp)
    st.write(response["response"])
