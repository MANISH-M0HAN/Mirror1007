import os
import logging
import time
from flask import request, jsonify, Blueprint
from dotenv import load_dotenv
from generate_response import get_response

load_dotenv()

question_bot_bp = Blueprint('question_bot_bp', __name__) 

@question_bot_bp.route('/question_chatbot', methods=['POST'])
def question_chatbot():
    try:
        start_time = time.time()
        recieved_api_key = request.headers.get("X-API-KEY")
        expected_api_key = os.getenv("API_KEY")

        if recieved_api_key != expected_api_key:
            return jsonify({"unauthorized_access": "invalid api key"}), 401

        user_input = request.json.get("user_input", "").strip()

        if not user_input:
            return jsonify({"error": "Missing user input"}), 400
        logging.info(f"Sent User Input: {user_input}")
        response = get_response(user_input)
        logging.info("Received Success Chat Agent Output")
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        send_time(total_time_ms)
        logging.info(f"Total time taken for request: {total_time_ms:.2f} ms")
        return jsonify({"response": response}), 200

    except Exception as exception:
        logging.info(f"Received Error Chat Agent Output:"
                     f"\n{str(exception)}")
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        send_time(total_time_ms)
        logging.info(f"Total time taken for request: {total_time_ms:.2f} ms")
        return jsonify({"error": str(exception)}), 500

def send_time(total_time_ms):
    file_path = "Debug/request_response_stack.csv"

    # Open the file and append the time taken to the last line
    with open(file_path, mode='r+', encoding='utf-8') as file:
        # Read all lines
        lines = file.readlines()

        # Check if there are lines to modify
        if lines:
            # Modify the last line by appending the time data
            lines[-1] = lines[-1].strip() + f", {total_time_ms:.2f} ms\n"

            # Move the cursor to the start and rewrite the modified lines
            file.seek(0)
            file.writelines(lines)
