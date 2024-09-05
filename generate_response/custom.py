import logging
from utils import process_user_input
from utils import load_prerequisites 
from utils import analyse_dataframe
from utils import domain_check
from utils import default_messages
from Tests.request_response_csv import request_response

def get_response(user_input, threshold=0.3):
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
        request_response(user_input, final_response)
        return final_response

    logging.info(f"After Spell Correction")
    corrected_input = process_user_input.correct_spelling(user_input)
    if corrected_input != user_input:
        logging.info(f"Corrected Input: {corrected_input}")
        context_response = analyse_dataframe.find_best_context(corrected_input, threshold)
        if context_response:
            column_response, best_match_response_flag = analyse_dataframe.match_columns(
                corrected_input, context_response
            )
            if best_match_response_flag == 1:
                column_response = (
                    column_response
                    + "\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."
                )
            request_response(corrected_input, column_response)
            return column_response

    logging.info(f"Checking Domain relevance")
    if domain_check.is_domain_relevant(corrected_input):
        prompt = f"User asked: {corrected_input}. Please provide a helpful response related to women's heart health."
        logging.info(f"Prompt for Generative API: {prompt}")
        response = default_messages.generate_response_with_placeholder(prompt)
        request_response(corrected_input, response)
        return response

    fallback_response = "I'm sorry, I can only answer questions related to women's heart health. Can you please clarify your question?"
    request_response(corrected_input, fallback_response)
    return fallback_response

logging.basicConfig(level=logging.INFO, filename='Debug/debug.log', filemode='a', format='%(asctime)s - %(message)s')

