import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot"


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers questions based on the provided context."),
        ("human", "Question: {question}"),
    ]
)
output_parser=StrOutputParser()
def generate_answer(question,llm_model,temperature,max_tokens):
    llm = ChatOllama(
        model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    chain = prompt | llm | output_parser
    response = chain.invoke({"question": question})
    return response

st.title("Simple Q&A Chatbot")
llm_model = st.sidebar.selectbox("Select LLM", ["llama3.2", "mistral","gemma3:1b"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 200)

st.write("Ask a question:")
user_input= st.text_input("Your question")
if user_input:  
    with st.spinner("Generating answer..."):
        answer = generate_answer(user_input, llm_model, temperature, max_tokens)
        st.write("Answer:", answer)