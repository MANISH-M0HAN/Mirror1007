import logging

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
