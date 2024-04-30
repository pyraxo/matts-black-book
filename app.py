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
from langchain.chains.query_constructor.base import AttributeInfo
# from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain import hub
from db import load_db, load_listings


load_dotenv()


text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

SYSTEM_PROMPT = (
    "You are a helpful AI car salesman. You are helping a user buy a car suited for them."
    "The provided context is a dataset containing a list of car models. "
    "The dataset includes the specific car model and the list of car reviews. "
    "Using the dataset of car models, suggest the most suitable 20 types of cars from the dataset for the user. "
    "You must reply with 20 types of cars, from the dataset."
    "For each of these, also include a summary of the reviews for each car, and an explanation of why it best fits the user query"
    "----------------"
    "{context}"
    "----------------"
)

SYSTEM_PROMPT2 = (
    "You are a helpful AI car salesman. You are helping a user buy a car suited for them."
    "The provided context is a set of current car listings from Craigslist. "
    "Using the dataset of car models, suggest the most suitable 5 cars from the dataset for the user. "
    "Provide a summarised version of the car listing, including the location, title, link and price of the car."
    "If you are unable to provide the specific cars listed, instead provide similar cars which can be found in the car listings."
    "Include the pros and cons of this car, specific to the user's requirements."
    "Also include brief summary of the reviews about the car."
    "You must reply with 5 car listings, from the Craigslist listings."
    "----------------"
    "{context}"
    "----------------"
)

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{input}")]
)

PROMPT2 = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT2), ("human", "{input}")]
)

# app = Flask(__name__)
# CORS(app)

metadata_field_info_db1 = [
    AttributeInfo(
        name="Vehicle_Title",
        description="The name and model of the car",
        type="string",
    ),
    AttributeInfo(
        name="Review",
        description="A series of reviews about the car. Separate reviews are delimited by ||",
        type="String",
    ),
]

document_content_description = "Set of car reviews"

metadata_field_info_db2 = [
    AttributeInfo(
        name="neighbourhood",
        description="The name and model of the car",
        type="string",
    ),
    AttributeInfo(
        name="post title",
        description="A series of reviews about the car. Separate reviews are delimited by ||",
        type="String",
    ),
    AttributeInfo(
        name="URL",
        description="A series of reviews about the car. Separate reviews are delimited by ||",
        type="String",
    ),
    AttributeInfo(
        name="price",
        description="A series of reviews about the car. Separate reviews are delimited by ||",
        type="Integer",
    ),
]
document_content_description2 = "Set of Craigslist listings"

# llm = ChatOpenAI(temperature=0)

# retriever = SelfQueryRetriever.from_llm(
#     llm,
#     vectorstore,
#     document_content_description,
#     metadata_field_info,
# )

# retriever = SelfQueryRetriever.from_llm(
#     llm,
#     self.load_db(),
#     document_content_description,
#     metadata_field_info,
#     search_kwargs={"k": 10},verbose=True
# )


def prompt(query):
    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    print(query)

    retriever = load_db()
    retriever2 = load_listings()
    # model = Ollama(base_url='http://119.74.32.2:11434',
    #                model="llama2", temperature=0.8)
    model = ChatOpenAI(model="gpt-4-turbo-2024-04-09",
                       api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

    # qachain = RetrievalQA.from_chain_type(model, retriever=retriever)

    question_answer_chain = create_stuff_documents_chain(
        model, PROMPT)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    question_answer_chain2 = create_stuff_documents_chain(
        model, PROMPT2)
    chain2 = create_retrieval_chain(retriever2, question_answer_chain2)

    output = chain.invoke({"input": query})
    output2 = chain2.invoke({"input": output["answer"]})
    # output = qachain.invoke({"query": prompt})
    print(output)
    print(output2)
    return {"response": output2["answer"]}


st.title("📓 Matt's Black Book")
if resp := st.text_area("Enter your query here", key="query"):
    response = prompt(resp)
    st.write(response["response"])
