import os
import pandas as pd
import logging
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker
from dotenv import load_dotenv
import nltk
from nltk.stem import WordNetLemmatizer

# Set the TOKENIZERS_PARALLELISM environment variable to avoid deadlock warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Download necessary data for lemmatization (only required once)
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'rand'  # Use a secure method to handle secret keys
CORS(app)

# Initialize models and spellchecker
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
spellchecker = SpellChecker()

# Load the CSV file into a DataFrame
csv_file = 'heart_health_triggers.csv' # Replace with the path to your CSV file
df = pd.read_csv(csv_file)
df.fillna('', inplace=True)

# Create a database list from the DataFrame
database = []
for index, row in df.iterrows():
    item = {
        "trigger_word": row['trigger_word'],
        "synonyms": row['synonyms'].split(','),  # Assuming synonyms are comma-separated
        "keywords": row['keywords'].split(','),  # Assuming keywords are comma-separated
        "What": row['What'],  # Response from 'What' column
        "Why": row['Why'],    # Response from 'Why' column
        "How": row['How'],    # Response from 'How' column
        "Symptoms": row['Symptoms']  # Response from 'Symptoms' column
    }
    database.append(item)

# Precompute embeddings for each question-related field in batches
trigger_embeddings = embedding_model.encode(df['trigger_word'].tolist(), batch_size=32)
synonyms_embeddings = [embedding_model.encode(syn.split(','), batch_size=32) for syn in df['synonyms']]
keywords_embeddings = [embedding_model.encode(kw.split(','), batch_size=32) for kw in df['keywords']]

# Precompute embeddings for column names
column_names = ['What', 'Why', 'How', 'Symptoms']
column_embeddings = embedding_model.encode(column_names)

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

def generate_response_with_placeholder(prompt):
    """
    Placeholder for the generative API response.
    Replaces the actual API call with a fixed response for development purposes.
    """
    response = "This is a placeholder response generated for your question."
    return response

def correct_spelling(text):
    """
    Corrects spelling errors in the given text using a spell checker.
    """
    if len(text.split()) > 1:
        corrected_words = [
            spellchecker.correction(word) if spellchecker.correction(word) else word
            for word in text.split()
        ]
        corrected_text = ' '.join(corrected_words)
        return corrected_text
    return text

def lemmatize_query(query):
    """
    Takes a query string and returns a lemmatized version of the query.
    """
    lemmatized_query = " ".join([lemmatizer.lemmatize(word) for word in query.split()])
    return lemmatized_query

def find_best_context(query, threshold):
    """
    Takes a query and returns the best matching context from the database based on direct matches with trigger, synonyms, or keywords.
    Uses average of max cosine similarity scores only if direct match is below a specified threshold.
    """
    query_embedding = embedding_model.encode([query.lower()])

    best_match_score = 0
    best_match_response = None
    best_match_type = None

    for index, item_embeddings in enumerate(db_embeddings):
        trigger_score = cosine_similarity(query_embedding, item_embeddings['trigger_embedding'].reshape(1, -1)).flatten()[0]
        synonym_scores = [cosine_similarity(query_embedding, syn_emb.reshape(1, -1)).flatten()[0] for syn_emb in item_embeddings['synonyms_embeddings']]
        keyword_scores = [cosine_similarity(query_embedding, kw_emb.reshape(1, -1)).flatten()[0] for kw_emb in item_embeddings['keywords_embeddings']]

        max_synonym_score = max(synonym_scores) if synonym_scores else 0
        max_keyword_score = max(keyword_scores) if keyword_scores else 0

        max_scores_sum = trigger_score + max_synonym_score + max_keyword_score
        avg_score = max_scores_sum / 3

        if trigger_score >= 0.65 or max_synonym_score >= 0.65 or max_keyword_score >= 0.65:
            logging.info(
                f"Strong direct match found. Query: '{query}', Trigger Score: {trigger_score:.4f}, "
                f"Synonym Score: {max_synonym_score:.4f}, Keyword Score: {max_keyword_score:.4f} "
                f"Response: {database[index]}"
            )
            return database[index]

        if avg_score > best_match_score and trigger_score < 0.65 and max_synonym_score < 0.65 and max_keyword_score < 0.65:
            best_match_score = avg_score
            best_match_response = database[index]
            best_match_type = "Avg Max Match"

    if best_match_score >= threshold:
        logging.info(
            f"Query: '{query}', Best Match Score: {best_match_score:.4f}, "
            f"Best Match Response: '{best_match_response}', Match Type: {best_match_type}"
        )
        return best_match_response
    else:
        logging.warning(f"No suitable match found for query: '{query}' with score above threshold: {threshold}")
        return None
    


def match_column(query, best_match_response):
    """
    Matches the query with the column name embeddings and returns the appropriate response from the best match row.
    Enhances accuracy by looking for specific keywords in the query.
    """
    
    query_lower = query.lower()
    query_lower = correct_spelling(query_lower) #Manish's function call for spell check should come here ~Myil
    
    if "what" in query_lower:
        return best_match_response['What']
    elif "why" in query_lower:
        return best_match_response['Why']
    elif "how" in query_lower:
        return best_match_response['How']
    elif "symptom" in query_lower or "sign" in query_lower:
        return best_match_response['Symptoms']

    query_embedding = embedding_model.encode([query_lower])
    column_scores = cosine_similarity(query_embedding, column_embeddings).flatten()

    best_column_index = column_scores.argmax()
    best_column_name = column_names[best_column_index]
    logging.info(f"Best column match (fallback): {best_column_name} with score {column_scores[best_column_index]:.4f}")

    return best_match_response[best_column_name]

def is_domain_relevant(query, threshold=0.4):
    """
    Checks if the query is relevant to the domain using cosine similarity.
    Returns True if any similarity score with domain keywords is above the threshold.
    """
    query_embedding = embedding_model.encode([query.lower()])
    relevance_scores = [cosine_similarity(query_embedding, [dom_emb]).flatten()[0] for dom_emb in domain_embeddings]

    logging.info(f"Domain Relevance Scores for '{query}': {relevance_scores}")

    return any(score >= threshold for score in relevance_scores)

def get_response(user_input, threshold=0.3):
    """
    Handles the logic to decide whether to use a pre-defined response or generate one with the API.
    Returns a response and updates the context history.
    """
    logging.info(f"Direct Match")
    context_response = find_best_context(user_input, threshold)
    if context_response:
        column_response = match_column(user_input, context_response)
        return column_response
    
    logging.info(f"After Spell Correction")
    corrected_input = correct_spelling(user_input)
    if corrected_input != user_input:
        logging.info(f"Corrected Input: {corrected_input}")
        context_response = find_best_context(corrected_input, threshold)
        if context_response:
            column_response = match_column(corrected_input, context_response)
            return column_response

    logging.info(f"Checking Domain relevance")
    if is_domain_relevant(corrected_input):
        prompt = f"User asked: {corrected_input}. Please provide a helpful response related to women's heart health."
        logging.info(f"Prompt for Generative API: {prompt}")
        response = generate_response_with_placeholder(prompt)
        return response 

    fallback_response = "I'm sorry, I can only answer questions related to women's heart health. Can you please clarify your question?"
    return fallback_response 

# Setup logging
logging.basicConfig(level=logging.INFO, filename='chatbot.log', filemode='a', format='%(asctime)s - %(message)s')

@app.route('/chatbot', methods=['POST'])
def chat():
    try:
        recieved_api_key = request.headers.get('X-API-KEY') 
        expected_api_key= 'fpv74NMceEzy.5OsNsX43uhfa2GSGPPOB1/o2ABXg0mMwniAef02'
        
        if recieved_api_key != expected_api_key:
            return jsonify({"unauthorized_access":"invalid api key"}), 401

        user_input = request.json.get("user_input", "").strip()

        if not user_input:
            return jsonify({"error": "Missing user input"}), 400

        response= get_response(user_input)

        return jsonify({"response": response}), 200

    except Exception as exception:
        logging.error(f"Error occurred: {str(exception)}")
        return jsonify({"error": str(exception)}), 500
if __name__ == '__main__':
    app.run(debug=True)
