import os
import google.generativeai as genai
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker
import numpy as np

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'randomtext'  # Replace with a random secret key for session management
CORS(app)

# Initialize models and spellchecker
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
spellchecker = SpellChecker()

# Configure Gemini API
genai.configure(api_key=os.getenv('GENERATIVE_AI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Dummy database for context retrieval (RAG concept)
database = [
    {"question": "What are the symptoms of heart disease?", "answer": "Symptoms of heart disease include chest pain, shortness of breath, and fatigue."},
    {"question": "How can I reduce high cholesterol?", "answer": "To reduce high cholesterol, eat a healthy diet, exercise regularly, and take prescribed medications."},
    {"question": "What is a healthy blood pressure range?", "answer": "A healthy blood pressure range is generally considered to be 120/80 mmHg."}
]
db_embeddings = [embedding_model.encode(item['question']) for item in database]

def correct_spelling(text):
    corrected_words = []
    for word in text.split():
        correction = spellchecker.correction(word)
        if correction and correction != word:
            corrected_words.append(correction)
        else:
            corrected_words.append(word)
    corrected_text = " ".join(corrected_words)
    return corrected_text

def find_best_context(query, threshold=0.4):
    query = correct_spelling(query)
    query_embedding = embedding_model.encode([query.lower()])
    similarities = cosine_similarity(query_embedding, db_embeddings).flatten()
    best_match_index = similarities.argmax()
    if similarities[best_match_index] < threshold:
        return None
    best_context = database[best_match_index]['answer']
    return best_context

def generate_response_with_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

def get_response(user_input, context_history, threshold=0.4):
    user_input = correct_spelling(user_input)
    greetings = ["hello", "hi", "hey"]

    if any(user_input.lower() == greeting for greeting in greetings):
        return "Hello! How can I assist you with your heart health questions today?", context_history

    context = find_best_context(user_input, threshold)
    if context:
        context_history.append(context)
    full_context = " ".join(context_history + [user_input])
    response = generate_response_with_gemini(full_context)

    context_history.append(response)
    if len(context_history) > 10:  # Limit context history to 10 exchanges
        context_history = context_history[-10:]

    return response, context_history

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get("user_input")
    if 'context_history' not in session:
        session['context_history'] = []
    context_history = session['context_history']
    response, context_history = get_response(user_input, context_history)
    session['context_history'] = context_history
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
