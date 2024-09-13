import os
import logging
from flask import Flask
from routes import question_bot_bp

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)
app.debug = True

app.register_blueprint(question_bot_bp)

logging.basicConfig(level=logging.INFO, filename='Debug/debug.log', filemode='a', format='%(asctime)s - %(message)s')

logging.info(f"Flask app Started!")
if __name__ == '__main__':
    app.run()

