import os
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker
import pandas as pd
import numpy as np

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'randomtext'  # Replace with a random secret key for session management
CORS(app)

# Initialize models and spellchecker
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
spellchecker = SpellChecker()

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

# Precompute embeddings for each question-related field
db_embeddings = []
for item in database:
    trigger_embedding = embedding_model.encode(item['trigger_word'])
    synonyms_embeddings = [embedding_model.encode(syn) for syn in item['synonyms']]
    keywords_embeddings = [embedding_model.encode(kw) for kw in item['keywords']]
    db_embeddings.append({
        "trigger_embedding": trigger_embedding,
        "synonyms_embeddings": synonyms_embeddings,
        "keywords_embeddings": keywords_embeddings
    })

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

def generate_fallback_response():
    return "I'm sorry, I don't have enough information to answer that. Could you please provide more details or ask a different question?"

def get_response(user_input, context_history, threshold=0.4):
    user_input = correct_spelling(user_input)
    greetings = ["hello", "hi", "hey"]

    if any(user_input.lower() == greeting for greeting in greetings):
        return "Hello! How can I assist you with your heart health questions today?", context_history

    context = find_best_context(user_input, threshold)
    if context:
        context_history.append(context)
        response = context
    else:
        response = generate_fallback_response()
    
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
