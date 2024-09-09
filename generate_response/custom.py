import logging
from utils import process_user_input
from utils import load_prerequisites 
from utils import analyse_dataframe
from utils import domain_check
from utils import default_messages
from Tests.request_response_csv import request_response
from utils import spell_checker

def get_response(user_input, threshold=0.3):
    word_set = spell_checker.load_word_set('./heart_health_triggers.csv', 
                             ['trigger_word', 'synonyms', 'keywords']) 
    logging.info(f"Before Spell Correct : {user_input}")
    user_input = spell_checker.correct_spelling(user_input, word_set)
    logging.info(f"After Spell Correct : {user_input}")

    logging.info(f"Direct Match")
    context_responses = analyse_dataframe.find_best_context(user_input, threshold)
    if context_responses:
        combined_responses = []

        for context_response in context_responses:
            # Fetch data from relevant columns
            column_response, best_match_response_flag = analyse_dataframe.match_columns(
                user_input, context_response
            )
            if column_response:
                combined_responses.append(column_response)

        # Combine all the column responses into a single response
        final_response = " \n\n ".join(combined_responses)
        if best_match_response_flag == 1:
            final_response = (
                final_response
                + "\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."
            )
        return final_response

    logging.info(f"Checking Domain relevance")
    if domain_check.is_domain_relevant(user_input):
        prompt = f"User asked: {user_input}. Please provide a helpful response related to women's heart health."
        logging.info(f"Prompt for Generative API: {prompt}")
        response = default_messages.generate_response_with_placeholder(prompt)
        request_response(user_input, response)
        return response

    fallback_response = "I'm sorry, I can only answer questions related to women's heart health. Can you please clarify your question?"
    request_response(user_input, fallback_response)
    return fallback_response

logging.basicConfig(level=logging.INFO, filename='Debug/debug.log', filemode='a', format='%(asctime)s - %(message)s')


