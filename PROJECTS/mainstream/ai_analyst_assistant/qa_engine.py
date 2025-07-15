import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def chunk_text(pages, chunk_size=400, overlap=100):
    chunks = []
    for page in pages:
        words = page["text"].split()
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append({
                "page": page["page"],
                "text": chunk
            })
            i += chunk_size - overlap
    return chunks

def embed_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
    return np.array(embeddings).astype('float32'), texts

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Use Cosine similarity
    index.add(embeddings)
    return index

def search_similar_chunks(index, question, texts, top_k=7):
    q_vec = model.encode([question], convert_to_tensor=False, normalize_embeddings=True).astype('float32')
    D, I = index.search(q_vec, top_k)
    return [texts[i] for i in I[0]]
