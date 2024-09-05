import os
from flask import Flask
from routes import question_bot_bp

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
app.debug = True

app.register_blueprint(question_bot_bp)

if __name__ == '__main__':
    app.run()

