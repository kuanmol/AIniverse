from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging
import re

model = SentenceTransformer("all-distilroberta-v1")

def chunk_text(pages, max_chunk_size=100):  # Smaller chunks for reports
    """Break text into smaller chunks for embedding, preserving structure."""
    chunks = []
    for page in pages:
        text = page["text"]
        # Split by sections or tables for better chunking
        sections = re.split(r'(?=\b(?:Introduction|Market Analysis|Data Analysis|Product Overview|Results & Discussion|Marketing Strategy|Sales Projections|Launch Timeline)\b)', text)
        for section in sections:
            words = section.split()
            for i in range(0, len(words), max_chunk_size):
                chunk = " ".join(words[i:i + max_chunk_size])
                if chunk.strip():
                    chunks.append(chunk)
    logging.info(f"Text split into {len(chunks)} chunks")
    return chunks

def embed_chunks(chunks):
    """Generate embeddings for text chunks."""
    try:
        embeddings = model.encode(chunks, show_progress_bar=False)
        logging.info("Chunks embedded successfully")
        return embeddings, chunks
    except Exception as e:
        logging.error(f"Embedding failed: {str(e)}")
        return np.array([]), []

def create_faiss_index(embeddings):
    """Create a FAISS index for similarity search."""
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        logging.info("FAISS index created")
        return index
    except Exception as e:
        logging.error(f"FAISS index creation failed: {str(e)}")
        return None

def search_similar_chunks(index, query, texts, top_k=3):  # Reduced for precision
    """Search for similar chunks to the query."""
    try:
        query_embedding = model.encode([query])[0]
        faiss.normalize_L2(np.array([query_embedding]))
        _, indices = index.search(np.array([query_embedding]), top_k)
        return [texts[i] for i in indices[0]]
    except Exception as e:
        logging.error(f"Search failed: {str(e)}")
        return []