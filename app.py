import os
import json
import torch
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from config import *

# =========================================================
# CONFIGURATION & CONSTANTS
# =========================================================
# Fallback token if not in environment
DEFAULT_HF_TOKEN = "NEW_HF_TOKEN"
HF_TOKEN = os.getenv("HF_TOKEN", DEFAULT_HF_TOKEN)

# Initialize FastAPI
app = FastAPI()

class Query(BaseModel):
    question: str

# =========================================================
# INITIALIZATION LOADER
# =========================================================
print("Initializing application...")

# 1. Login
login(token=HF_TOKEN)

# 2. Load Resources (Assume files exist as per user request)
print("Loading existing RAG documents and Index...")
if not os.path.exists(RAG_DOCS_PATH):
    print(f"Warning: {RAG_DOCS_PATH} not found. Please ensure it exists.")
else:
    with open(RAG_DOCS_PATH, "r", encoding="utf-8") as f:
        rag_docs = json.load(f)

if not os.path.exists(NEW_FAISS_INDEX_PATH):
    print(f"Warning: {NEW_FAISS_INDEX_PATH} not found. Please ensure it exists.")
    index = None
else:
    # Load the index - FAISS handles the type automatically
    index = faiss.read_index(NEW_FAISS_INDEX_PATH)
    # Note: User specified HNSW parameters, but read_index reconstructs the index structure.
    # If parameters need tuning for search time, they can be set on index.hnsw object here.
    # e.g., index.hnsw.efSearch = 50 
    if hasattr(index, 'hnsw'):
        index.hnsw.efSearch = 50

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Load LLM
print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

# =========================================================
# RAG LOGIC
# =========================================================
def retrieve_top_k(query, k=5):
    if index is None:
        return []
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding).astype("float32"), k)
    retrieved_docs = [rag_docs[i] for i in I[0]]
    return retrieved_docs

def rag_answer(question, k=3):
    docs = retrieve_top_k(question, k)
    context = "\n\n".join(d["text"] for d in docs)

    prompt = f"""
You are a concise assistant.

Answer the question using ONLY the context below.
Use at most TWO short sentences.
Do NOT mention the context or the question.
If the answer is not found, say: I don't know.

Context:
{context}

Answer:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            temperature=0.2
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract Answer
    answer = full_output.split("Answer:")[-1].strip()
    
    # Enforce max 2 sentences logic from rag_bot.py
    sentences = answer.split(".")
    answer = ".".join(sentences[:2]).strip()
    if not answer.endswith("."):
        answer += "."
        
    return answer

# =========================================================
# API ENDPOINT
# =========================================================
@app.post("/ask")
def ask_question(q: Query):
    try:
        answer = rag_answer(q.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================
# CLI ENTRY POINT
# =========================================================
if __name__ == "__main__":
    import uvicorn
    print("\n✅ NewApp initialized with existing index. Starting CLI loop...")
    print("To run as API server: uvicorn newapp:app --reload\n")
    
    while True:
        try:
            q = input("\n❓ Ask a question (or type 'exit'): ")
            if q.lower() in ["exit", "quit"]:
                break
            answer = rag_answer(q)
            print(answer)
        except KeyboardInterrupt:
            break
