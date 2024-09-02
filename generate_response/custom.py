import logging
from utils import process_user_input
from utils import load_prerequisites 
from utils import analyse_dataframe
from utils import domain_check
from utils import default_messages

def get_response(user_input, threshold=0.3):
    logging.info(f"Direct Match")
    context_response = analyse_dataframe.find_best_context(user_input, threshold)
    if context_response:
        # Fetch data from relevant columns
        column_response = analyse_dataframe.match_columns(user_input, context_response)
        return column_response
    
    logging.info(f"After Spell Correction")
    corrected_input = process_user_input.correct_spelling(user_input)
    if corrected_input != user_input:
        logging.info(f"Corrected Input: {corrected_input}")
        context_response = analyse_dataframe.find_best_context(corrected_input, threshold)
        if context_response:
            column_response = analyse_dataframe.match_columns(corrected_input, context_response)
            return column_response

    logging.info(f"Checking Domain relevance")
    if domain_check.is_domain_relevant(corrected_input):
        prompt = f"User asked: {corrected_input}. Please provide a helpful response related to women's heart health."
        logging.info(f"Prompt for Generative API: {prompt}")
        response = default_messages.generate_response_with_placeholder(prompt)
        return response 

    fallback_response = "I'm sorry, I can only answer questions related to women's heart health. Can you please clarify your question?"
    return fallback_response



logging.basicConfig(level=logging.INFO, filename='chatbot.log', filemode='a', format='%(asctime)s - %(message)s')

