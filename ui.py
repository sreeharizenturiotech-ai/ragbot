import streamlit as st
import requests

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 RAG Chatbot")
st.write("Ask questions from your knowledge base")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(
                BACKEND_URL,
                json={"question": question}
            )

            if response.status_code == 200:
                answer = response.json().get("answer")
                st.success("Answer:")
                st.write(answer)
            else:
                st.error("Backend error")
