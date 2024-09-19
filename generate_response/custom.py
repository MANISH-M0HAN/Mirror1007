import logging
from utils import (
    default_messages,
    process_user_input,
    load_prerequisites,
    analyse_dataframe,
    domain_check,
    spell_checker
)
from Tests.request_response_csv import request_response

def get_response(raw_user_input, threshold=0.3):
    #Load word set for spell checking
    word_set = spell_checker.load_word_set('./heart_health_triggers.csv', 
                             ['trigger_word', 'synonyms', 'keywords']) 
    
    #Log the raw input before and after spell correction
    logging.info(f"Before Spell Correct : {raw_user_input}")
    user_input = spell_checker.correct_spelling(raw_user_input, word_set)
    logging.info(f"After Spell Correct : {user_input}")
    
    # Direct match search
    logging.info(f"Direct Match")
    context_responses = analyse_dataframe.find_best_context(user_input, threshold)
    
    if context_responses:
        combined_responses = []

        # Fetch data from relevant columns
        for context_response in context_responses:
            column_response, best_match_response_flag = analyse_dataframe.match_columns(user_input, context_response)
            if column_response:
                combined_responses.append(column_response)

        # Combine all the column responses into a single response
        final_response = " \n\n ".join(combined_responses)
        if best_match_response_flag == 1:
            final_response = ( final_response + "\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs.")
            
        # Log the request and response  
        request_response(raw_user_input, user_input, final_response)
        return final_response
    
    #Check for domain relevance
    logging.info(f"Checking Domain relevance")
    if domain_check.is_domain_relevant(user_input):
        prompt = f"User asked: {user_input}. Please provide a helpful response related to women's heart health."
        logging.info(f"Prompt for Generative API: {prompt}")
        response = default_messages.generate_response_with_placeholder(prompt)
        
        # Log the request and response
        request_response(raw_user_input, user_input, response)
        return response
    
    #Fallback response for irrelevant quries
    fallback_response = "I'm sorry, I can only answer questions related to women's heart health. Can you please clarify your question?"
    request_response(raw_user_input, user_input, fallback_response)
    return fallback_response

#Configure logging setting
logging.basicConfig(
    level=logging.INFO,
    filename='Debug/debug.log', 
    filemode='a',
    format='%(asctime)s - %(message)s'
    )


