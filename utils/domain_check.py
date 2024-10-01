from utils.load_prerequisites import embedding_model, domain_embeddings, domain_keywords
from sklearn.metrics.pairwise import cosine_similarity
import logging

def is_domain_relevant(query, threshold=0.6):
    logging.info(f"At Domain Relevance. Threshold to pass domain relevancy is {threshold}. The query is '{query}'")
    
    query_embedding = embedding_model.encode([query.lower()])
    relevance_scores = [
        cosine_similarity(query_embedding, [dom_emb]).flatten()[0] 
        for dom_emb in domain_embeddings
    ]
    
    total_relevance_score = sum(relevance_scores)
    avg_relevance_score = total_relevance_score / len(relevance_scores) if relevance_scores else 0
    
    max_domain_score = max(relevance_scores)
    max_index = relevance_scores.index(max_domain_score)
    max_domain_word = domain_keywords[max_index]
    
    logging.info(f"Highest domain score: {max_domain_score:.4f} for keyword '{max_domain_word}'")
    logging.info(f"Total relevance score: {total_relevance_score:.4f}, Average relevance score: {avg_relevance_score:.4f}")
    
    # Return based on average relevance
    if avg_relevance_score >= threshold:
        logging.info(f"Query '{query}' is domain-relevant (Avg Score: {avg_relevance_score:.4f})")
        return True
    else:
        logging.info(f"Query '{query}' is not domain-relevant (Avg Score: {avg_relevance_score:.4f})")
        return False