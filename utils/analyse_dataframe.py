from sklearn.metrics.pairwise import cosine_similarity
from utils import load_prerequisites 
import logging

def match_generator(query_words):
    for index, item_embeddings in enumerate(load_prerequisites.db_embeddings):
        trigger_words = [word.lower().strip() for trigger in load_prerequisites.database[index]["trigger_words"] for word in trigger.split(' ')]
        synonyms = [word.lower().strip() for synonym in load_prerequisites.database[index]["synonyms"]for word in synonym.split(' ')]
        keywords = [word.lower().strip() for keyword in load_prerequisites.database[index]["keywords"]for word in keyword.split(' ')]
        
        all_match_words = set(trigger_words + synonyms + keywords)
        
        common_words = all_match_words & set([word.lower().strip() for word in query_words])
        logging.debug(f"Common words found: {common_words}")
        if common_words:
            logging.warning(f"Yielding database entry: {load_prerequisites.database[index]}")
            yield load_prerequisites.database[index]

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

def evaluate_matches(avg_match_score, max_match_score, avg_match_response, max_match_response, avg_match_count, max_match_count, avg_match_flag, max_match_flag, threshold):
    if avg_match_score >= threshold and max_match_score < avg_match_score:
        logging.info(
            f"Avg Match Score: {avg_match_score:.4f}, Avg Match Response: '{avg_match_response}"
            f"Match Count: {avg_match_count}, Avg Match: {avg_match_flag}"
        )
        print("This is from Average Match Response")
        return avg_match_response

    elif max_match_score > avg_match_score:
        logging.info(
            f"Max Match Score: {max_match_score:.4f}, Max Match Response: '{max_match_response}'" 
            f"Match Count: {max_match_count}, Max Match: {max_match_flag}"
        )
        print("This is from Max Match Response")
        return max_match_response

    else:
        logging.warning(
            f"No suitable match found for query with score above threshold: {threshold}"
        )
        return None

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
    
def match_columns(query, matched_response):
    ambiguous_query_flag = 0
    
    intent_words = {
            'What': ['what', 'defin', 'identif', 'describ', 'clarif', 'specif', 'detail', 'outlin', 'state', 'explain', 'determin', 'depict', 'summar', 'design', 'distinguish', 'mean'], 
            'Symptoms': ['symptom', 'sign', 'indic', 'manifest', 'warn', 'clue', 'evid', 'redflag', 'marker', 'present', 'outcom', 'pattern', 'phenomena', 'trait', 'occurr'], 
            'Why': ['why', 'caus', 'reason', 'purpos', 'explain', 'justif', 'origin', 'motiv', 'trigger', 'rational', 'ground', 'basi', 'excus', 'sourc', 'factor'], 
            'How': ['how', 'method', 'mean', 'procedur', 'step', 'techniqu', 'process', 'way', 'approach', 'strateg', 'system', 'manner', 'framework', 'mode', 'prevent', 'avoid', 'safeguard', 'protect', 'mitig', 'reduct', 'intervent', 'defen', 'deter', 'shield', 'do']
        }
    
    matching_columns = []
    match_found = False 

    for column, keywords in intent_words.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            position = query.find(keyword_lower)
            if position != -1 and matched_response.get(column):
                snippet_start = max(0, position - 8)
                snippet_end = min(len(query), position + len(keyword_lower) + 8)
                snippet = query[snippet_start:snippet_end]
                logging.info(f"Match found for column: {column} with keyword: '{keyword}' at position {position}. Snippet: '{snippet}'")
                matching_columns.append((position, matched_response[column]))
                match_found = True  
                break  

    if not match_found and intent_words:
        first_column = next(iter(intent_words)) 
        if matched_response.get(first_column):
            default_response = matched_response[first_column]
            ambiguous_query_flag = 1
            logging.info(f"No match found. Fetching default response : {default_response} from column :{first_column}.")
            matching_columns.append((0, default_response))

    matching_columns.sort(key=lambda x: x[0])

    responses = [response for _, response in matching_columns]

    if responses:
        return " ".join(responses), ambiguous_query_flag
