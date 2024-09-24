import logging
from utils import load_prerequisites
from .score_matches import score_matches
from .evaluate_matches import evaluate_matches
from .match_generator import match_generator


def find_best_context(query, threshold):
    query = query.split()
    query_embedding = load_prerequisites.embedding_model.encode([' '.join(query)])
    logging.info(f"1)Direct Match for all Trigger, Synonym and Keywords")
    matches = list(match_generator(query))
    if matches:
        return matches

    (avg_match_score, max_match_score, 
    avg_match_response, max_match_response,
    avg_match_count, max_match_count,
    avg_match_flag, max_match_flag) = score_matches(query_embedding)
     
    return evaluate_matches(
        avg_match_score, max_match_score, 
        avg_match_response, max_match_response, 
        avg_match_count, max_match_count,
        avg_match_flag, max_match_flag,
        threshold
    )

