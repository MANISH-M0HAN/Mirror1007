from flask import jsonify

def unauthorized_user_error():
    unauthorized_error_json = {
        "data": {
            "chatbot-response": "",
            "bot-type": ""
        },
        "message": "Unauthorised request. The API key provided is not valid.",
        "status": "Error"
    }

    return jsonify(unauthorized_error_json), 401

def success_response(chatbot_response, bot_type= None):
    if bot_type== None:
        bot_type = "Custom"
    success_response_json = {
        "data": {
            "chatbot-response": chatbot_response,
            "bot-type": bot_type
        },
        "message": "Successfully recieved response",
        "status": "Success"
    }

    return jsonify(success_response_json), 200

def validation_error(message, bot_type= None):
    if bot_type== None:
        bot_type = "Custom"
    validation_error_json = {
        "data": {
            "chatbot-response": message,
            "bot-type": bot_type
        },
        "message": "Validation error",
        "status": "Error"
    }

    return jsonify(validation_error_json), 422

def internal_server_error(exception):
    internal_server_error_json = {
        "data": {
            "chatbot-response": "",
            "bot-type": ""
        },
        "message": "Internal server error",
        "status": "Error"
    }

    return jsonify(internal_server_error_json), 500
