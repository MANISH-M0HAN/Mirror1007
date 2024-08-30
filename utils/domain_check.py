def is_domain_relevant(query, threshold=0.4):
    query_embedding = embedding_model.encode([query.lower()])
    relevance_scores = [cosine_similarity(query_embedding, [dom_emb]).flatten()[0] for dom_emb in domain_embeddings]

    logging.info(f"Domain Relevance Scores for '{query}': {relevance_scores}")

    return any(score >= threshold for score in relevance_scores)
