import logging
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_prerequisites

def score_matches(query_embedding):
    avg_match_score = 0
    max_match_score = 0
    avg_match_response = []
    max_match_response = []
    avg_match_count = 0
    max_match_count = 0
    avg_match_flag = False
    max_match_flag = False

    logging.info(f"2)Cosine Match with Avg Score or Max")
    for index, item_embeddings in enumerate(load_prerequisites.db_embeddings):
        trigger_scores = [
            cosine_similarity(query_embedding, trw_emb.reshape(1, -1)).flatten()[0]
            for trw_emb in item_embeddings["triggers_embeddings"]
        ]
        synonym_scores = [
            cosine_similarity(query_embedding, syn_emb.reshape(1, -1)).flatten()[0]
            for syn_emb in item_embeddings["synonyms_embeddings"]
        ]
        keyword_scores = [
            cosine_similarity(query_embedding, kw_emb.reshape(1, -1)).flatten()[0]
            for kw_emb in item_embeddings["keywords_embeddings"]
        ]

        max_trigger_score = max(trigger_scores) if trigger_scores else 0
        max_synonym_score = max(synonym_scores) if synonym_scores else 0
        max_keyword_score = max(keyword_scores) if keyword_scores else 0

        max_scores_sum = max_trigger_score + max_synonym_score + max_keyword_score
        avg_score = max_scores_sum / 3

        if (
            max_trigger_score >= 0.65
            or max_synonym_score >= 0.65
            or max_keyword_score >= 0.65
        ):
            logging.info(
                f"Strong direct match found for one of the features."
                f"Trigger Score: {max_trigger_score:.4f} Synonym Score: {max_synonym_score:.4f}, Keyword Score: {max_keyword_score:.4f}"
                f"Response: {load_prerequisites.database[index]}"
            )
            max_match_score = max(max_trigger_score, max_synonym_score, max_keyword_score)
            max_match_response.append(load_prerequisites.database[index])
            max_match_count += 1
            max_match_flag = True

        if (
            avg_score > avg_match_score
            and max_trigger_score < 0.65
            and max_synonym_score < 0.65
            and max_keyword_score < 0.65
        ):
            logging.info(
                f"Strong Average match found. Avg Score: {avg_score:.4f},"
                f"Trigger Score: {max_trigger_score:.4f}, Synonym Score: {max_synonym_score:.4f},"
                f"Keyword Score: {max_keyword_score:.4f} Response: {load_prerequisites.database[index]}"
            )
            avg_match_score = avg_score
            avg_match_response.append(load_prerequisites.database[index])
            avg_match_count += 1
            avg_match_flag = True

    return (avg_match_score, max_match_score, 
            avg_match_response, max_match_response, 
            avg_match_count, max_match_count, 
            avg_match_flag, max_match_flag)
