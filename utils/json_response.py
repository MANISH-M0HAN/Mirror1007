from flask import jsonify

def unauthorized_user_error():
    unauthorized_error_json = {
        "data": {
            "chatbot-response": "",
            "bot-type": ""
        },
        "message": "Unauthorised Request",
        "status": "error"
    }

    return jsonify(unauthorized_error_json),401

def bot_response(response, bot_type= None):
    if bot_type== None:
        bot_type = "custom"
    bot_response_json = {
        "data": {
            "chatbot-response": response,
            "bot-type": bot_type
        },
        "message": "Sucessfully recieved response",
        "status": "success"
    }

    return jsonify(bot_response_json),200

def validation_error(message, bot_type= None):
    if bot_type== None:
        bot_type = "custom"
    validation_error_json = {
        "data": {
            "chatbot-response": message,
            "bot-type": bot_type
        },
        "message": "validation error",
        "status": "error"
    }

    return jsonify(validation_error_json),422
