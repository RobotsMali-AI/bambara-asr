import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union

# --- Define the Document Embedding Model ---
DOCUMENT_MODEL_NAME = "dangvantuan/french-document-embedding"
SENTENCE_MODEL_NAME = "dangvantuan/sentence-camembert-base"

# Dictionary to cache the loaded models
MODEL_CACHE = {}

def load_embedding_model(model_name: str):
    """Loads and caches the specified SentenceTransformer model."""
    if model_name not in MODEL_CACHE:
        try:
            # We must use 'trust_remote_code=True' for this specific model
            model = SentenceTransformer(model_name, trust_remote_code=True)
            MODEL_CACHE[model_name] = model
            print(f"✅ Loaded embedding model: {model_name}")
        except Exception as e:
            raise RuntimeError(f"❌ Error loading model {model_name}: {e}")
    return MODEL_CACHE[model_name]

def calculate_cosine_similarity(
    text_a: str, 
    text_b: str, 
    use_document_model: bool = False
) -> float:
    """
    Calculates the cosine similarity between two texts, selecting the appropriate 
    model based on the 'use_document_model' flag.
    """
    
    # 1. Select the model based on the flag
    model_name = DOCUMENT_MODEL_NAME if use_document_model else SENTENCE_MODEL_NAME
    model = load_embedding_model(model_name)
    
    if not text_a or not text_b:
        return 0.0
    
    # 2. Encode the texts
    embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
    
    emb_a = embeddings[0].reshape(1, -1)
    emb_b = embeddings[1].reshape(1, -1)
    
    # 3. Calculate the cosine similarity
    similarity_score = cosine_similarity(emb_a, emb_b)[0][0]

    # Cosine similarity typically ranges from -1 (opposite) to 1 (identical).
    # Since the Sentence-Transformers models are often trained contrastively 
    # to cluster similar items closer, scores are usually between 0 and 1 
    # for semantic search/clustering tasks.
    
    return float(similarity_score)