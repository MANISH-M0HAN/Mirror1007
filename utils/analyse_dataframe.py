from sklearn.metrics.pairwise import cosine_similarity
from utils import load_prerequisites 
import logging
from .intent_words import intent_words


def prepare_query(query):
    query_embedding = load_prerequisites.embedding_model.encode([query.lower()])
    query_words = query.strip().lower().split()
    return query_embedding, query_words

