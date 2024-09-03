import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "rand")
    API_KEY = os.getenv("API_KEY")
    CSV_FILE = os.getenv("CSV_FILE", "data/heart_health_triggers.csv")

