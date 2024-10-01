import os
import logging
import time
from flask import request, jsonify, Blueprint
from dotenv import load_dotenv
from generate_response import get_response
from utils.json_response import unauthorized_user_error, success_response, validation_error, internal_server_error

load_dotenv()

question_bot_bp = Blueprint('question_bot_bp', __name__) 

@question_bot_bp.route('/question_chatbot', methods=['POST'])
def question_chatbot():
    try:
        start_time = time.time()
        recieved_api_key = request.headers.get("X-API-KEY")
        expected_api_key = os.getenv("API_KEY")
        
        if recieved_api_key != expected_api_key:
            return unauthorized_user_error()
          
        user_input = request.json.get("user-input", "").strip()
        
        if not user_input:
            message = "Missing user input"
            return validation_error(message)
          
        logging.info(f"Sent User Input: {user_input}")
        
        custom_response = get_response(user_input)
        logging.info("Received Success Chat Agent Output")
        
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        send_time(total_time_ms)
        logging.info(f"Total time taken for request: {total_time_ms:.2f} ms")
        
        return success_response(custom_response)

    except Exception as exception:
        logging.info("Error Message:"
                     f"\n{str(exception)}")
        exception = str(exception)
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        send_time(total_time_ms)
        logging.info(f"Total time taken for request: {total_time_ms:.2f} ms")
        return internal_server_error(exception)

def send_time(total_time_ms):
    file_path = "Debug/request_response_stack.csv"
    with open(file_path, mode='r+', encoding='utf-8') as file:
        lines = file.readlines()
        if lines:
            lines[-1] = lines[-1].strip() + f", {total_time_ms:.2f} ms\n"
            file.seek(0)
            file.writelines(lines)