import os
import json
import uuid
import numpy as np
import torch
import faiss

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

from config import *

# ---------- AUTH ----------
# set token once in terminal:
# export HF_TOKEN=your_token  (Linux/Mac)
# setx HF_TOKEN your_token    (Windows)

login(token=os.getenv("HF_TOKEN"))

# ---------- CHUNKING ----------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks


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


# ---------- DATA PROCESS ----------
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


# ---------- BUILD RAG ----------
os.makedirs("outputs", exist_ok=True)

rag_docs = []
rag_docs += process_youtube_data(YOUTUBE_PATH)
rag_docs += process_website_data(WEBSITE_PATH)

with open(RAG_DOCS_PATH, "w", encoding="utf-8") as f:
    json.dump(rag_docs, f, indent=2, ensure_ascii=False)

# ---------- EMBEDDINGS ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
texts = [d["text"] for d in rag_docs]
embeddings = embedder.encode(texts, show_progress_bar=True).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)

# ---------- LLM ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

# ---------- RAG QUERY ----------
def retrieve(query, k=5):
    q_emb = embedder.encode([query]).astype("float32")
    _, ids = index.search(q_emb, k)
    return [rag_docs[i] for i in ids[0]]

def ask(question):
    context = "\n\n".join(d["text"] for d in retrieve(question))

    prompt = f"""
Answer using ONLY this context.
Max 2 sentences.
If unknown, say: I don't know.

Context:
{context}

Answer:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=80)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("Answer:")[-1].strip()


# ---------- CLI ----------
while True:
    q = input("\n❓ Ask (exit to quit): ")
    if q.lower() == "exit":
        break
    print(ask(q))
