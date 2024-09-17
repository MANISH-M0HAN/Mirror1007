import logging
from utils import analyse_dataframe
from utils import domain_check
from utils import default_messages
from Tests.request_response_csv import request_response
from utils import spell_checker

def get_response(raw_user_input, threshold=0.3):
    word_set = spell_checker.load_word_set('./heart_health_triggers.csv', 
                             ['trigger_word', 'synonyms', 'keywords']) 
    logging.info(f"Before Regex : {raw_user_input}")
    regexd_user_input = analyse_dataframe.preprocess_input(raw_user_input)
    logging.info(f"After Regex : {regexd_user_input}")
    logging.info(f"Before Spell Correct : {regexd_user_input}")
    spell_corrected_user_input = spell_checker.correct_spelling(regexd_user_input, word_set)
    logging.info(f"After Spell Correct : {spell_corrected_user_input}")
    
    context_responses = analyse_dataframe.find_best_context(spell_corrected_user_input, threshold)
    if context_responses:
        combined_responses = []

        for context_response in context_responses:
            logging.info(f"Row is picked, now triggering match_columns()")
            column_response, ambiguous_query_flag = analyse_dataframe.match_columns(spell_corrected_user_input, context_response)
            if column_response:
                combined_responses.append(column_response)

        final_response = " \n\n ".join(combined_responses)
        if ambiguous_query_flag == 1:
            final_response = (
                final_response
                + "\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."
            )
        request_response(raw_user_input, spell_corrected_user_input, final_response)
        return final_response
    
    logging.info(f"3)Checking Domain relevance")
    if domain_check.is_domain_relevant(spell_corrected_user_input):
        logging.info(f"Passing to AI method")
        prompt = f"User asked: {spell_corrected_user_input}. Please provide a helpful response related to women's heart health."
        logging.info(f"Prompt for Generative API: {prompt}")
        response = default_messages.generate_response_with_placeholder(prompt)
        request_response(raw_user_input, spell_corrected_user_input, response)
        return response
    logging.info(f"Failed Domain relevance")
    fallback_response = "I'm sorry, I can only answer questions related to women's heart health. Can you please clarify your question?"
    request_response(raw_user_input, spell_corrected_user_input, fallback_response)
    return fallback_response


