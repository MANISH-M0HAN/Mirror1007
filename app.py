import os
from flask import Blueprint, Flask, request, jsonify, session
from routes import chatbot

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
app.debug = True

chatbot= app.BluePrint('chatbot app', __name__)

if __name__ == '__main__':
    app.run()

