import logging
from utils import load_prerequisites 

def match_generator(query_words):
    for index, item_embeddings in enumerate(load_prerequisites.db_embeddings):
        trigger_words = [trigger.lower().strip() for trigger in load_prerequisites.database[index]["trigger_words"]]
        synonyms = [synonym.lower().strip() for synonym in load_prerequisites.database[index]["synonyms"]]
        keywords = [keyword.lower().strip() for keyword in load_prerequisites.database[index]["keywords"]]

        all_match_words = set(trigger_words + synonyms + keywords)
        common_words = all_match_words & set([word.lower().strip() for word in query_words])
        logging.debug(f"Common words found: {common_words}")
        
        if common_words:
            logging.warning(f"Yielding database entry: {load_prerequisites.database[index]}")
            yield load_prerequisites.database[index]
