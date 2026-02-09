import os
import json
import torch
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

from fastapi import FastAPI
from pydantic import BaseModel

from config import *

# =========================================================
# FASTAPI INIT
# =========================================================
app = FastAPI()

class Query(BaseModel):
    question: str

# =========================================================
# AUTHENTICATION
# =========================================================
login(token=os.getenv("HF_TOKEN"))

# =========================================================
# LOAD EXISTING RAG DOCUMENTS + FAISS INDEX
# =========================================================
with open(RAG_DOCS_PATH, "r", encoding="utf-8") as f:
    rag_docs = json.load(f)

index = faiss.read_index(FAISS_INDEX_PATH)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =========================================================
# LOAD LLM
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

# =========================================================
# RETRIEVAL + GENERATION
# =========================================================
def retrieve_top_k(query, k=5):
    query_embedding = embedder.encode([query]).astype("float32")
    _, indices = index.search(query_embedding, k)
    return [rag_docs[i] for i in indices[0]]

def rag_answer(question, k=5):
    docs = retrieve_top_k(question, k)
    context = "\n\n".join(d["text"] for d in docs)

    prompt = f"""
You are a concise assistant.

Answer the question using ONLY the context below.
Use at most TWO short sentences.
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
            do_sample=False
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_output.split("Answer:")[-1].strip()
    return answer

# =========================================================
# FASTAPI ENDPOINT
# =========================================================
@app.post("/ask")
def ask_question(q: Query):
    answer = rag_answer(q.question)
    return {"answer": answer}

# =========================================================
# CLI TEST
# =========================================================
if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        print(rag_answer(q))
