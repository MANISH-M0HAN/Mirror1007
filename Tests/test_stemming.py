import requests
import os
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("CHATBOT_URL")
api_key = os.getenv("API_KEY")

headers = {
    "X-API-KEY": api_key,
    "Content-Type": "application/json"
}

def get_bot_response(user_input):
    try:
        payload = {"user_input": user_input}
        response = requests.post(url, headers=headers, json=payload)
        return response.json().get("response", "")
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return ""

what_response = "Angina is chest pain caused by reduced blood flow to the heart."
why_response = "It signals an increased risk of heart attack."
how_response = "Manage stress, avoid heavy meals, and follow a healthy lifestyle."
symptoms_response = "Chest pain, shortness of breath, nausea."
intent_words = {
    "What": [
        "What", "Define", "Identify", "Describe", "Clarify", "Specify", "Detail", "Outline", "State", "Explain", "Determine", 
        "Depict", "Summarize", "Designat", "Distinguish"
    ],
    "Symptoms": [
        "Symptoms", "Signs", "Indications", "Manifestations", "Warning", "Clues", "Evidence", "Redflags", "Markers", 
        "Presentations", "Outcomes", "Patterns", "Phenomena", "Traits", "Occurrences"
    ],
    "Why": [
        "Why", "Causes", "Reason", "Purpose", "Explain", "Justification", "Origin", "Motive", "Trigger", "Rationale", 
        "Grounds", "Basis", "Excuse", "Source", "Factor"
    ],
    "How": [
        "How", "Method", "Means", "Procedure", "Steps", "Technique", "Process", "Way", "Approach", "Strategy", "System", 
        "Manner", "Framework", "Form", "Mode", "Prevention", "Avoidance", "Safeguard", "Protection", "Mitigation", 
        "Reduction", "Intervention", "Defense", "Deterrence", "Shielding", "Do"
    ]
}


trigger_words = ["angina"]


def generate_queries(intent_words, trigger_words):
    query_combinations = []
    
    for intent_category, words in intent_words.items():
        for word in words:
            for trigger in trigger_words:
                if intent_category == "What":
                    query = f"{word} {trigger}?"
                    response = get_bot_response(query)
                    if response != what_response:
                        print(response, "\n", word, "\n",intent_category)
                elif intent_category == "Symptoms":
                    query = f"{word} {trigger}?"
                    response = get_bot_response(query)
                    if response != symptoms_response:
                        print(response, "\n", word, "\n",intent_category)
                elif intent_category == "Why":
                    query = f"{word} {trigger}?"
                    response = get_bot_response(query)
                    if response != why_response:
                        print(response, "\n", word, "\n",intent_category)
                elif intent_category == "How":
                    query = f"{word} {trigger}?"
                    response = get_bot_response(query)
                    if response != how_response:
                        print(response, "\n", word, "\n",intent_category)
                
                query_combinations.append(query)
    
    return query_combinations

queries = generate_queries(intent_words, trigger_words)



