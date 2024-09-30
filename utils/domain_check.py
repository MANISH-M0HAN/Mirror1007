from utils.load_prerequisites import embedding_model, domain_embeddings, domain_keywords
from sklearn.metrics.pairwise import cosine_similarity
import logging

def is_domain_relevant(query, threshold=0.8):
    logging.info(f"At Domain Relevance. Threshold to pass domain relavancy is {threshold}. "
                 f"The query is {query}")
    query_embedding = embedding_model.encode([query.lower()])
    relevance_scores = [cosine_similarity(query_embedding, [dom_emb]).flatten()[0] for dom_emb in domain_embeddings]

    max_domain_score = max(relevance_scores)
    max_index = relevance_scores.index(max_domain_score)
    max_domain_word = domain_keywords[max_index]

    logging.info(f"The highest domain score is {max_domain_score:.4f} for keyword '{max_domain_word}'")
    return any(score >= threshold for score in relevance_scores)
