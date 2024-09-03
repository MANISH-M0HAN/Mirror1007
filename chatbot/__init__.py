from flask import Flask
from flask_cors import CORS
from .config import Config

# Initialize the Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Import the routes after initializing the app
from . import routes

