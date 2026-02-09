import streamlit as st
from app import rag_answer  # adjust if your function name differs

st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("🤖 RAG Chatbot")
st.write("Ask questions based on your documents")

question = st.text_input("Enter your question:")

if question:
    with st.spinner("Thinking..."):
        answer, _ = rag_answer(question)
    st.success(answer)
