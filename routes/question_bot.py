import os
import logging
import time
from flask import request, jsonify, Blueprint
from dotenv import load_dotenv
from generate_response import get_response
from utils.json_response import unauthorized_user_error, bot_response, validation_error, internal_server_error

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

        user_input = request.json.get("user_input", "").strip()

        if not user_input:
            message = "Missing user input"
            return validation_error(message)

        response = get_response(user_input)

        return bot_response(response)

    except Exception as exception:
        message = str(exception)
        return internal_server_error(message)


