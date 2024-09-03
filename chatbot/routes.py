from flask import request, jsonify
from . import app
from .response_generation import get_response
import logging

@app.route("/chatbot", methods=["POST"])
def chat():
    try:
        recieved_api_key = request.headers.get("X-API-KEY")
        expected_api_key = app.config["API_KEY"]

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

