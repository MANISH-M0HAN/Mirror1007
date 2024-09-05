import os
import logging
from flask import request, jsonify, Blueprint
from dotenv import load_dotenv
from generate_response import get_response

load_dotenv()

question_bot_bp = Blueprint('question_bot_bp', __name__) 

@question_bot_bp.route('/question_chatbot', methods=['POST'])
def question_chatbot():
    try:
        recieved_api_key = request.headers.get("X-API-KEY")
        expected_api_key = os.getenv("API_KEY")

        if recieved_api_key != expected_api_key:
            return jsonify({"unauthorized_access": "invalid api key"}), 401

        user_input = request.json.get("user_input", "").strip()

        if not user_input:
            return jsonify({"error": "Missing user input"}), 400

        response = get_response(user_input)

        return jsonify({"response": response}), 200

    except Exception as exception:
        logging.error(f"Error occurred: {str(exception)}")
        return jsonify({"error": str(exception)}), 500


