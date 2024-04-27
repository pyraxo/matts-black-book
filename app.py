import os
from dotenv import load_dotenv
from flask_cors import CORS
from flask import Flask, request

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
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise."
    "{context}"
)

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{input}")]
)

app = Flask(__name__)
CORS(app)


@app.route('/prompt', methods=['POST'])
def prompt():
    # Get prompt from request data
    query = request.json['query']
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


if __name__ == "__main__":
    load_db()
    app.run(port=8000, debug=True)
