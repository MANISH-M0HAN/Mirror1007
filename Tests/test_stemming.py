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
        response.raise_for_status()  # Ensure request was successful
        return response.json().get("response", "")
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return ""
    
expected_responses = {
    "What": "Angina is chest pain caused by reduced blood flow to the heart.",
    "Symptoms": "Chest pain, shortness of breath, nausea.",
    "Why": "It signals an increased risk of heart attack.",
    "How": "Manage stress, avoid heavy meals, and follow a healthy lifestyle."
}

intent_words = {
    "What": ["What", "Define", "Identify", "Describe", "Clarify", "Specify", "Detail", "Outline", "State", "Explain", "Determine", 
             "Depict", "Summarize", "Designat", "Distinguish"],
    "Symptoms": ["Symptoms", "Signs", "Indications", "Manifestations", "Warning", "Clues", "Evidence", "Redflags", "Markers", 
                 "Presentations", "Outcomes", "Patterns", "Phenomena", "Traits", "Occurrences"],
    "Why": ["Why", "Causes", "Reason", "Purpose", "Explain", "Justification", "Origin", "Motive", "Trigger", "Rationale", 
            "Grounds", "Basis", "Excuse", "Source", "Factor"],
    "How": ["How", "Method", "Means", "Procedure", "Steps", "Technique", "Process", "Way", "Approach", "Strategy", "System", 
            "Manner", "Framework", "Form", "Mode", "Prevention", "Avoidance", "Safeguard", "Protection", "Mitigation", 
            "Reduction", "Intervention", "Defense", "Deterrence", "Shielding", "Do"]
}

trigger_words = ["angina"]

def generate_queries(intent_words, trigger_words, expected_responses):
    query_combinations = []
    
    for intent, words in intent_words.items():
        expected_response = expected_responses.get(intent, "")
        
        for word in words:
            for trigger in trigger_words:
                query = f"{word} {trigger}?"
                response = get_bot_response(query)
                
                if response != expected_response:
                    print(f"Query: {query}\nResponse: {response}\nExpected: {expected_response}\nIntent: {intent}\nWord: {word}")
                
                query_combinations.append(query)
    
    return query_combinations

queries = generate_queries(intent_words, trigger_words, expected_responses)