import os
import pandas as pd
import logging
from flask import Flask, request, jsonify
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
nltk.download("wordnet")
nltk.download("omw-1.4")

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
            best_max_match_score = max(trigger_score, max_synonym_score, max_keyword_score)
            best_max_match_response = database[index]
            best_max_match_type = "Max Match"
        
        if avg_score > best_match_score and trigger_score < 0.65 and max_synonym_score < 0.65 and max_keyword_score < 0.65:
            best_match_score = avg_score
            best_match_response = database[index]
            best_match_type = "Avg Max Match"
            
    if best_match_score >= threshold and best_max_match_score < best_match_score:
        return best_match_response
    
    elif best_max_match_score > best_match_score:
        return best_max_match_response
    
    else:
        return None
    
    
def match_columns(query, best_match_response):
    query_lower = query.lower()
    query_lower = correct_spelling(query_lower)

    # Dictionary of intent words with prioritized order
    intent_words = {       
        "What": [
            "What", "Define", "Identify", "Describe", "Clarify", "Specify", "Detail", "Outline", "State",
            "Explain", "Determine", "Depict", "Summarize", "Designate", "Distinguish", "Relate", "Relationship",
        ],
        "Why": [
            "Why", "Causes", "Reason", "Purpose", "Explain", "Justification", "Origin", "Motive", "Trigger",
            "Rationale", "Grounds", "Basis", "Excuse", "Source", "Factor",
        ],
        "How": [
            "How", "Method", "Means", "Procedure", "Steps", "Technique", "Process", "Way", "Approach",
            "Strategy", "System", "Manner", "Framework", "Form", "Mode", "Prevention", "Avoidance",
            "Safeguard", "Protection", "Mitigation", "Reduction", "Intervention", "Defense", "Deterrence",
            "Shielding", "Manage", "Treatment",
        ],
        "Symptoms": [
            "Symptoms", "Signs", "Indications", "Manifestations", "Warning", "Clues", "Evidence", "Redflags",
            "Markers", "Presentations", "Outcomes", "Patterns", "Phenomena", "Traits", "Occurrences",
        ],
    }

    # Collect matching columns and their first occurrence positions
    matching_columns = []
    for column, keywords in intent_words.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            position = query_lower.find(keyword_lower)
            if position != -1 and best_match_response.get(column):
                matching_columns.append((position, column, best_match_response[column]))
                break  # Move to the next column once a match is found

    # Log matched columns for debugging
    logging.info(f"Matched Columns: {matching_columns}")

    # Sort the matched columns by the position of their first occurrence in the query
    matching_columns.sort(key=lambda x: x[0])

    # Separate responses into definitions and prevention measures
    definitions = []
    prevention = []
    for _, column, response in matching_columns:
        if column == "What":
            definitions.append(response)
        elif column == "How":
            prevention.append(response)

    # Log categorized responses for debugging
    logging.info(f"Definitions: {definitions}")
    logging.info(f"Prevention: {prevention}")

    # Handle specific information about relationships between terms if multiple triggers are present
    if "cholesterol" in query_lower and "tachycardia" in query_lower:
        relationship_statement = "Cholesterol levels can affect heart health, including conditions like tachycardia. Managing cholesterol is crucial for heart health."
    else:
        relationship_statement = ""

    # Combine responses with definitions first and prevention measures second
    combined_responses = definitions
    if relationship_statement:
        combined_responses.append(relationship_statement)
    combined_responses.extend(prevention)

    # Ensure we do not repeat the same response
    unique_responses = list(dict.fromkeys(combined_responses))

    # Return the combined response
    combined_response = " ".join(unique_responses)

    # Log final response for debugging
    logging.info(f"Combined Response: {combined_response}")

    return combined_response if combined_response else "I'm sorry, I couldn't find a relevant answer."

    
def get_response(user_input, threshold=0.3):
    logging.info(f"Processing user input: {user_input}")
    
    # Step 1: Attempt to find a best match context based on the user input
    context_response = find_best_context(user_input, threshold)
    if context_response:
        logging.info("Direct match found in context.")
        
        # Fetch and combine data from relevant columns based on the identified context
        column_response = match_columns(user_input, context_response)
        return column_response
    
    # Step 2: Perform spell correction if no direct match is found
    corrected_input = correct_spelling(user_input)
    if corrected_input != user_input:
        logging.info(f"Spell corrected input: {corrected_input}")
        
        context_response = find_best_context(corrected_input, threshold)
        if context_response:
            logging.info("Match found after spell correction.")
            
            # Fetch and combine data from relevant columns based on the corrected input
            column_response = match_columns(corrected_input, context_response)
            return column_response
    
    # Step 3: Check if the query is domain relevant
    if is_domain_relevant(corrected_input):
        logging.info("Input is domain relevant.")
        
        # Generate a response using a placeholder generative API
        prompt = f"User asked: {corrected_input}. Please provide a helpful response related to women's heart health."
        logging.info(f"Prompt for Generative API: {prompt}")
        
        response = generate_response_with_placeholder(prompt)
        return response
    
    # Step 4: Fallback response if no matches are found
    fallback_response = "I'm sorry, I can only answer questions related to women's heart health. Can you please clarify your question?"
    logging.info("No relevant match found. Providing fallback response.")
    return fallback_response



logging.basicConfig(level=logging.INFO, filename='chatbot.log', filemode='a', format='%(asctime)s - %(message)s')


@app.route('/chatbot', methods=['POST'])
def chat():
    try:
        received_api_key = request.headers.get('X-API-KEY') 
        expected_api_key = 'fpv74NMceEzy.5OsNsX43uhfa2GSGPPOB1/o2ABXg0mMwniAef02'
        
        if received_api_key != expected_api_key:
            return jsonify({"unauthorized_access": "invalid api key"}), 401

        user_input = request.json.get("user_input", "").strip()

        if not user_input:
            return jsonify({"error": "Missing user input"}), 400

        response = get_response(user_input)
        return jsonify({"response": response}), 200

    except Exception as exception:
        logging.error(f"Error occurred: {str(exception)}")
        return jsonify({"error": str(exception)}), 500

if __name__ == '__main__':
    app.run(debug=True)




