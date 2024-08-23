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
    response = "This is a placeholder response generated for your question."
    return response

def correct_spelling(text):
    if len(text.split()) > 1:
        corrected_words = [
            spellchecker.correction(word) if spellchecker.correction(word) else word
            for word in text.split()
        ]
        corrected_text = ' '.join(corrected_words)
        return corrected_text
    return text

def lemmatize_query(query):
    lemmatized_query = " ".join([lemmatizer.lemmatize(word) for word in query.split()])
    return lemmatized_query

def find_best_context(query, threshold):
    query_embedding = embedding_model.encode([query.lower()])

    best_match_score = 0
    best_max_match_score = 0
    best_match_response = None
    best_max_match_response = None
    best_match_type = None
    best_max_match_type = None

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
            best_max_match_score = max(trigger_score, max_synonym_score, max_keyword_score)
            best_max_match_response = database[index]
            best_max_match_type = "Max Match"
        
        if avg_score > best_match_score and trigger_score < 0.65 and max_synonym_score < 0.65 and max_keyword_score < 0.65:
            logging.info(
                f"Strong direct match found. Query: '{query}', Trigger Score: {trigger_score:.4f}, "
                f"Synonym Score: {max_synonym_score:.4f}, Keyword Score: {max_keyword_score:.4f} "
                f"Response: {database[index]}"
            )
            best_match_score = avg_score
            best_match_response = database[index]
            best_match_type = "Avg Max Match"
            
    if best_match_score >= threshold and best_max_match_score < best_match_score:
        logging.info(
            f"Query: '{query}', Best Match Score: {best_match_score:.4f}, "
            f"Best Match Response: '{best_match_response}', Match Type: {best_match_type}"
        )
        return best_match_response
    
    elif best_max_match_score > best_match_score:
        logging.info(
            f"Query: '{query}', Max Match Score: {best_max_match_score:.4f}, "
            f"Best Match Response: '{best_max_match_response}', Match Type: {best_max_match_type}"
        )
        return best_max_match_response
    
    else:
        logging.warning(f"No suitable match found for query: '{query}' with score above threshold: {threshold}")
        return None

def match_columns(query, best_match_response):
    query_lower = query.lower()
    query_lower = correct_spelling(query_lower) 
    
    
    intent_words = {
        "Symptoms": [
            "Symptoms", "Signs", "Indications", "Manifestations", 
            "What are the symptoms", "What signs", "What does it feel like", 
            "How does it manifest", "What are the warning signs", 
            "What could indicate", "What happens when", "How does it show", 
            "What’s the symptomatology"
        ],
        "Why": [
            "Why", "For what reason", "How come", "causes", "What causes", "Why is it that", 
            "Why do", "Why does", "Why should", "Explain why", "Give the reason", 
            "What’s the purpose of", "What’s the point of", "Why do you think", 
            "What’s the reason for"
        ],
        "How": [
            "How", "In what way", "By what means", "How do", "How does", "How to", 
            "How can", "How might", "How could", "Explain how", "Describe how", 
            "In what manner", "In what method", "What steps", "What’s the procedure for"
        ],
        "What": [
            "What", "Which", "Identify", "Define", "Explain", "Describe", "Clarify",
            "Tell me about", "What is", "What are", "What's", "What exactly"
        ]
    }

    # Collect responses from matching columns
    responses = []
    for column, keywords in intent_words.items():
        if any(keyword in query_lower for keyword in keywords):
            if column in best_match_response:
                responses.append(best_match_response[column])

    # Combine responses from multiple columns without column names
    if responses:
        return " ".join(responses)

    query_embedding = embedding_model.encode([query_lower])
    column_scores = cosine_similarity(query_embedding, column_embeddings).flatten()

    best_column_index = column_scores.argmax()
    best_column_name = column_names[best_column_index]
    logging.info(f"Best column match (fallback): {best_column_name} with score {column_scores[best_column_index]:.4f}")

    #return best_match_response[best_column_name]

def match_columns(query, best_match_response):
    query_lower = query.lower()
    query_lower = correct_spelling(query_lower) 

    intent_words = {
        "Symptoms": [
            "Symptoms", "Signs", "Indications", "Manifestations", 
            "What are the symptoms", "What signs", "What does it feel like", 
            "How does it manifest", "What are the warning signs", 
            "What could indicate", "What happens when", "How does it show", 
            "What’s the symptomatology"
        ],
        "Why": [
            "Why", "For what reason", "How come", "causes", "What causes", 
            "Why is it that", "Why do", "Why does", "Why should", "Explain why", 
            "Give the reason", "What’s the purpose of", "What’s the point of", 
            "Why do you think", "What’s the reason for"
        ],
        "How": [
            "How", "In what way", "By what means", "How do", "How does", "How to", 
            "How can", "How might", "How could", "Explain how", "Describe how", 
            "In what manner", "In what method", "What steps", "What’s the procedure for"
        ],
        "What": [
            "What", "Which", "Identify", "Define", "Explain", "Describe", "Clarify",
            "Tell me about", "What is", "What are", "What's", "What exactly"
        ]
    }

    # Collect responses from matching columns
    responses = []
    for column, keywords in intent_words.items():
        if any(keyword.lower() in query_lower for keyword in keywords):
            if best_match_response.get(column):
                responses.append(best_match_response[column])

    # If responses from multiple columns are found, concatenate them
    if responses:
        return " ".join(responses)

    # Fallback to the best matching column if no intent word is matched
    query_embedding = embedding_model.encode([query_lower])
    column_scores = cosine_similarity(query_embedding, column_embeddings).flatten()

    best_column_index = column_scores.argmax()
    best_column_name = column_names[best_column_index]
    logging.info(f"Best column match (fallback): {best_column_name} with score {column_scores[best_column_index]:.4f}")

    return best_match_response.get(best_column_name, "")


def is_domain_relevant(query, threshold=0.4):
    query_embedding = embedding_model.encode([query.lower()])
    relevance_scores = [cosine_similarity(query_embedding, [dom_emb]).flatten()[0] for dom_emb in domain_embeddings]

    logging.info(f"Domain Relevance Scores for '{query}': {relevance_scores}")

    return any(score >= threshold for score in relevance_scores)

def get_response(user_input, threshold=0.3):
    logging.info(f"Direct Match")
    context_response = find_best_context(user_input, threshold)
    if context_response:
        # Fetch data from relevant columns
        column_response = match_columns(user_input, context_response)
        return column_response
    
    logging.info(f"After Spell Correction")
    corrected_input = correct_spelling(user_input)
    if corrected_input != user_input:
        logging.info(f"Corrected Input: {corrected_input}")
        context_response = find_best_context(corrected_input, threshold)
        if context_response:
            column_response = match_columns(corrected_input, context_response)
            return column_response

    logging.info(f"Checking Domain relevance")
    if is_domain_relevant(corrected_input):
        prompt = f"User asked: {corrected_input}. Please provide a helpful response related to women's heart health."
        logging.info(f"Prompt for Generative API: {prompt}")
        response = generate_response_with_placeholder(prompt)
        return response 

    fallback_response = "I'm sorry, I can only answer questions related to women's heart health. Can you please clarify your question?"
    return fallback_response



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
