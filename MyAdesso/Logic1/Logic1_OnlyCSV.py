import os
import pandas as pd
import logging
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker
from dotenv import load_dotenv

# Set the TOKENIZERS_PARALLELISM environment variable to avoid deadlock warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'randomtext'  # Use a secure method to handle secret keys

# Initialize CORS with specific configuration
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
    """
    Corrects spelling errors in the given text using a spell checker.
    """
    corrected_words = [
        spellchecker.correction(word) if spellchecker.correction(word) else word
        for word in text.split()
    ]
    corrected_text = ' '.join(corrected_words)
    return corrected_text

def find_best_context(query, threshold=0.4):
    """
    Takes a query and returns the best matching context from the database based on cosine similarity.
    If no match is found above the threshold, it returns None.
    """
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

def generate_response_with_placeholder(prompt):
    """
    Placeholder for the generative API response.
    Replaces the actual API call with a fixed response for development purposes.
    """
    response = "This is a placeholder response generated for your question."
    return response

def is_domain_relevant(query):
    """
    Checks if the query contains keywords relevant to the domain (women's heart health).
    """
    domain_keywords = ['heart', 'cardiac', 'women', 'health', 'cardiology']
    query_tokens = set(query.lower().split())
    return any(keyword in query_tokens for keyword in domain_keywords)

def get_relevant_context(user_input, context_history):
    """
    Retrieves relevant context from the context history based on the user input.
    Maintains coherence in conversation by using recent history.
    """
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
    """
    Handles the logic to decide whether to use a pre-defined response or generate one with the API.
    Returns a response and updates the context history.
    """
    # Direct match with original input
    context_response = find_best_context(user_input, threshold)
    if context_response:
        return context_response, context_history

    # Correct spelling and try matching again
    corrected_input = correct_spelling(user_input)
    context_response = find_best_context(corrected_input, threshold)
    if context_response:
        return context_response, context_history

    # Check for domain relevance
    if is_domain_relevant(corrected_input):
        prompt = f"User asked: {corrected_input}. Please provide a helpful response related to women's heart health."
        logging.info(f"Prompt for Generative API: {prompt}")
        
        # Send corrected input to the Generative API
        response = generate_response_with_placeholder(prompt)
        return response, context_history

    # Fallback response for unrelated queries
    fallback_response = "I'm sorry, I can only answer questions related to women's heart health. Can you please clarify your question?"
    return fallback_response, context_history

# Setup logging
logging.basicConfig(level=logging.INFO, filename='chatbot.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """
    Main endpoint for handling user input and returning the chatbot response.
    Uses session to maintain context history.
    """
    try:
        # Check if session is new, send initial greeting
        if 'context_history' not in session:
            session['context_history'] = {"context": [], "history": []}
            initial_greeting = "Hello! How can I assist you today?"
            return jsonify({"response": initial_greeting})

        if not request.is_json:
            return jsonify({"error": "Invalid input"}), 400

        data = request.get_json()
        user_input = data.get('user_input')
        context_history = session['context_history']

        if user_input is None:
            return jsonify({"error": "Missing 'user_input' parameter"}), 400

        logging.info(f"User input: {user_input}")

        response, context_history = get_response(user_input, context_history)
        session['context_history'] = context_history

        logging.info(f"Response: {response}")

        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"Error handling request: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/session', methods=['GET'])
def get_session():
    """
    Endpoint to retrieve session data, useful for debugging and ensuring context is being maintained.
    """
    session_data = {key: value for key, value in session.items()}
    return jsonify({"session_data": session_data})

if __name__ == "__main__":
    app.run(debug=True)
