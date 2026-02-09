import os
import json
import uuid
import numpy as np
import torch
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

from config import *

# =========================================================
# AUTHENTICATION
# =========================================================
# Set token once in terminal:
# Windows: setx HF_TOKEN your_token
# Linux/Mac: export HF_TOKEN=your_token

login(token=os.getenv("HF_TOKEN"))

# =========================================================
# TEXT CHUNKING
# =========================================================
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks


# =========================================================
# LOAD JSON FILES
# =========================================================
def load_json_files(path):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.load(f)]

    data = []
    for file in os.listdir(path):
        if file.endswith(".json"):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                data.append(json.load(f))
    return data


# =========================================================
# DATA PROCESSING
# =========================================================
def process_youtube_data(path):
    docs = []
    for item in load_json_files(path):
        for seg in item.get("segments", []):
            for i, chunk in enumerate(chunk_text(seg.get("text", ""))):
                docs.append({
                    "id": str(uuid.uuid4()),
                    "source": "youtube",
                    "text": chunk,
                    "metadata": {
                        "video_id": item.get("video_id"),
                        "chunk_id": i
                    }
                })
    return docs


def process_website_data(path):
    docs = []
    for item in load_json_files(path):
        for i, chunk in enumerate(chunk_text(item.get("page_text", ""))):
            docs.append({
                "id": str(uuid.uuid4()),
                "source": "website",
                "text": chunk,
                "metadata": {
                    "page_url": item.get("page_url"),
                    "chunk_id": i
                }
            })
    return docs


# =========================================================
# BUILD RAG DOCUMENTS + FAISS INDEX (ONCE)
# =========================================================
os.makedirs("outputs", exist_ok=True)

rag_docs = []
rag_docs.extend(process_youtube_data(YOUTUBE_PATH))
rag_docs.extend(process_website_data(WEBSITE_PATH))

with open(RAG_DOCS_PATH, "w", encoding="utf-8") as f:
    json.dump(rag_docs, f, indent=2, ensure_ascii=False)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

texts = [doc["text"] for doc in rag_docs]
embeddings = embedder.encode(texts, show_progress_bar=True).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)

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
    answer = full_output.split("Answer:")[-1].strip()

    sentences = answer.split(".")
    answer = ".".join(sentences[:2]).strip()
    if not answer.endswith("."):
        answer += "."

    return answer


# =========================================================
# CLI ENTRY POINT (SAFE)
# =========================================================
if __name__ == "__main__":
    while True:
        question = input("\n❓ Ask a question (or type 'exit'): ")
        if question.lower() in ["exit", "quit"]:
            break
        print(rag_answer(question))
