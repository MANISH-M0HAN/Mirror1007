'''
This code is working good for case where i ask 
w
k
women
etc in query.
Wrong things in this code is 
1. Session Endpoint is wrong, it is creating while i wanted to just check what is stored in session
2. Too many log, easy to remove them
3. I want to make bot have capability to differentiate "I am PMS" with "What is PMS"
4. Check for more issues with this code
'''
import os
import pandas as pd
import logging
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'rand'  # Use a secure method to handle secret keys
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

# Precompute embeddings for domain keywords
domain_keywords = ['heart', 'cardiac', 'women', 'health', 'cardiology']
domain_embeddings = embedding_model.encode(domain_keywords)

def correct_spelling(text):
    """
    Corrects spelling errors in the given text using a spell checker.
    """
    # Correct only for longer texts or obvious typos
    if len(text.split()) > 1:
        corrected_words = [
            spellchecker.correction(word) if spellchecker.correction(word) else word
            for word in text.split()
        ]
        corrected_text = ' '.join(corrected_words)
        return corrected_text
    return text

def find_best_context(query, threshold=0.7):
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

        # Logging the similarity scores for analysis
        logging.info(f"Query: {query}, Trigger: {database[index]['trigger_word']}, Score: {max_score}")

        if max_score > best_match_score and max_score >= threshold:
            best_match_score = max_score
            best_match_response = database[index]['response']

    # Log the best match score and response
    logging.info(f"Best Match Score: {best_match_score}, Best Match Response: {best_match_response}")

    return best_match_response

def generate_response_with_placeholder(prompt):
    """
    Placeholder for the generative API response.
    Replaces the actual API call with a fixed response for development purposes.
    """
    response = "This is a placeholder response generated for your question."
    return response

def is_domain_relevant(query, threshold=0.4):
    """
    Checks if the query is relevant to the domain using cosine similarity.
    Returns True if any similarity score with domain keywords is above the threshold.
    """
    query_embedding = embedding_model.encode([query.lower()])
    relevance_scores = [cosine_similarity(query_embedding, [dom_emb]).flatten()[0] for dom_emb in domain_embeddings]

    # Log the domain relevance scores
    logging.info(f"Domain Relevance Scores for '{query}': {relevance_scores}")

    return any(score >= threshold for score in relevance_scores)

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

def get_response(user_input, context_history, threshold=0.7):
    """
    Handles the logic to decide whether to use a pre-defined response or generate one with the API.
    Returns a response and updates the context history.
    """
    # Direct match with original input
    context_response = find_best_context(user_input, threshold)
    if context_response:
        return context_response, context_history

    # Correct spelling and try matching again (skip correction for very short inputs)
    corrected_input = correct_spelling(user_input)
    if corrected_input != user_input:
        logging.info(f"Corrected Input: {corrected_input}")
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
logging.basicConfig(level=logging.INFO, filename='chatbot.log', filemode='a', format='%(asctime)s - %(message)s')

@app.route('/chatbot', methods=['POST'])
def chat():
    """
    Main chat endpoint to handle user requests.
    Accepts user input and returns the chatbot's response.
    """
    try:
        user_input = request.json.get("user_input", "").strip()
        
        # Retrieve or initialize context history
        context_history = session.get('context_history', {'history': []})

        if not user_input:
            return jsonify({"error": "Missing user input"}), 400

        # Get response and updated context history
        response, updated_context_history = get_response(user_input, context_history)

        # Append the new interaction to the history
        updated_context_history['history'].append({
            "user_input": user_input,
            "bot_response": response
        })

        # Update the session with new context history
        session['context_history'] = updated_context_history

        return jsonify({"response": response}), 200

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({"error": "An error occurred"}), 500

@app.route('/session', methods=['POST'])
def create_session():
    """
    Endpoint to create a new session for the user.
    Initializes a fresh context history.
    """
    session['context_history'] = {'history': []}
    return jsonify({"message": "Session created"}), 200

if __name__ == '__main__':
    app.run(debug=True)
