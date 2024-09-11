import os
import logging
import time
from flask import request, jsonify, Blueprint
from dotenv import load_dotenv
from generate_response import get_response
<<<<<<< HEAD
<<<<<<< HEAD
from utils.json_response import unauthorized_user_error, success_response, validation_error, internal_server_error
=======
from utils.json_response import unauthorized_user_error, bot_response, validation_error
>>>>>>> 1986c76 (feat: create json response methods)
=======
from utils.json_response import unauthorized_user_error, success_response, validation_error, internal_server_error
>>>>>>> ca65cda (refactor: change function names)

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

        custom_response = get_response(user_input)

<<<<<<< HEAD
        return success_response(custom_response)
=======
        return success_response(response)
>>>>>>> ca65cda (refactor: change function names)

    except Exception as exception:
        exception = str(exception)
        return internal_server_error(exception)


