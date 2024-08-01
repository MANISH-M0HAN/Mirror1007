import os
import google.generativeai as genai
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker
import pandas as pd
import numpy as np
from dotenv import load_dotenv
# from gingerit.gingerit import GingerIt

import logging

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = os.getenv('secret_key')
api_key = os.getenv('GENERATIVE_AI_API_KEY')
CORS(app)

# Initialize models and spellchecker
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
spellchecker = SpellChecker()

# Configure Gemini API
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Load the CSV file into a DataFrame
csv_file = 'heart_health_triggers.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)
df.fillna('', inplace=True)

# Create a database list from the DataFrame
database = []
for index, row in df.iterrows():
    item = {
        "trigger_word": row['trigger_word'],
        "synonyms": row['synonyms'].split(','),  # Assuming synonyms are comma-separated
        "keywords": row['keywords'].split(','),  # Assuming keywords are comma-separated
        "response": row['response']
    }
    database.append(item)

# Precompute embeddings for each question-related field in batches
trigger_embeddings = embedding_model.encode(df['trigger_word'].tolist(), batch_size=32)
synonyms_embeddings = [embedding_model.encode(syn.split(','), batch_size=32) for syn in df['synonyms']]
keywords_embeddings = [embedding_model.encode(kw.split(','), batch_size=32) for kw in df['keywords']]

db_embeddings = []
for idx in range(len(df)):
    db_embeddings.append({
        "trigger_embedding": trigger_embeddings[idx],
        "synonyms_embeddings": synonyms_embeddings[idx],
        "keywords_embeddings": keywords_embeddings[idx]
    })

def correct_spelling(text):
    parser = GingerIt()
    result = parser.parse(text)
    return result['result']

def find_best_context(query, threshold=0.4):
    query = correct_spelling(query)
    query_embedding = embedding_model.encode([query.lower()])
    
    best_match_score = 0
    best_match_response = None
    
    for index, item_embeddings in enumerate(db_embeddings):
        trigger_score = cosine_similarity(query_embedding, [item_embeddings['trigger_embedding']]).flatten()[0]
        synonyms_scores = [cosine_similarity(query_embedding, [syn_emb]).flatten()[0] for syn_emb in item_embeddings['synonyms_embeddings']]
        keywords_scores = [cosine_similarity(query_embedding, [kw_emb]).flatten()[0] for kw_emb in item_embeddings['keywords_embeddings']]
        
        max_synonym_score = max(synonyms_scores) if synonyms_scores else 0
        max_keyword_score = max(keywords_scores) if keywords_scores else 0
        
        max_score = max(trigger_score, max_synonym_score, max_keyword_score)
        
        if max_score > best_match_score and max_score >= threshold:
            best_match_score = max_score
            best_match_response = database[index]['response']
    
    return best_match_response

def generate_response_with_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt, max_length=150)
        if response and response['choices']:
            return response['choices'][0]['text'].strip()
        else:
            return "I'm sorry, but I couldn't generate a response. Please try rephrasing your question."
    except Exception as e:
        return f"Error: {e}"

def get_relevant_context(user_input, context_history):
    follow_up_keywords = ["what about", "and", "also", "more", "else", "further"]

    if any(keyword in user_input.lower() for keyword in follow_up_keywords) and len(context_history['history']) > 1:
        relevant_context = " ".join(
            [f"User: {h['user_input']} Bot: {h['bot_response']}" for h in context_history['history'][-2:]]
        )
    else:
        relevant_context = " ".join(
            [f"User: {h['user_input']} Bot: {h['bot_response']}" for h in context_history['history'][-1:]]
        )

    return relevant_context

def get_response(user_input, context_history, threshold=0.4):
    user_input = correct_spelling(user_input)
    greetings = ["hello", "hi", "hey"]

    if any(user_input.lower() == greeting for greeting in greetings):
        return "Hello! How can I assist you with your heart health questions today?", context_history

    context = find_best_context(user_input, threshold)
    if context:
        context_history['context'].append({"info": context})
    context_history['history'].append({"user_input": user_input, "bot_response": ""})

    relevant_context = get_relevant_context(user_input, context_history)
    prompt = f"User asked: {user_input}\nContext: {relevant_context}\nPlease provide a concise response within 150 words."
    response = generate_response_with_gemini(prompt)

    context_history['history'][-1]['bot_response'] = response
    if len(context_history['history']) > 10:  # Limit context history to 10 exchanges
        context_history['history'] = context_history['history'][-10:]

    return response, context_history

# Setup logging
logging.basicConfig(level=logging.INFO, filename='chatbot.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get("user_input")
    if 'context_history' not in session:
        session['context_history'] = {"context": [], "history": []}
    context_history = session['context_history']

    logging.info(f"User input: {user_input}")

    response, context_history = get_response(user_input, context_history)
    session['context_history'] = context_history

    logging.info(f"Response: {response}")

    return jsonify({"response": response})

@app.route('/context_history', methods=['GET'])
def get_context_history():
    context_history = session.get('context_history', {"context": [], "history": []})
    return jsonify({"context_history": context_history})

@app.route('/clear_context_history', methods=['POST'])
def clear_context_history():
    session.pop('context_history', None)
    return jsonify({"message": "Context history cleared."})

@app.route('/verify_session', methods=['GET'])
def verify_session():
    session_data = {key: value for key, value in session.items()}
    return jsonify({"session_data": session_data})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
