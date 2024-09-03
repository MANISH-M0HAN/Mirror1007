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

# Set the Tokenizers_parallelism enviroment variable to avoid deadlock warning 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#Load enviromemnt variables from .env file
load_dotenv()

#Download necessary data for lemmatization (only required once)
nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = "rand"  # Use a secure method to handle secret keys
CORS(app)

# Initialize models and spellchecker
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
spellchecker = SpellChecker()

# Load and the preprocess CSV data
csv_file = 'heart_health_triggers.csv' # Replace with the path to your CSV file
df = pd.read_csv(csv_file).fillna('')

# Create a database list from the DataFrame
database = []
for index, row in df.iterrows():
    item = {
        "trigger_word": row["trigger_word"],
        "synonyms": row["synonyms"].split(","),  # Assuming synonyms are comma-separated
        "keywords": row["keywords"].split(","),  # Assuming keywords are comma-separated
        "What": row["What"],  # Response from 'What' column
        "Why": row["Why"],  # Response from 'Why' column
        "How": row["How"],  # Response from 'How' column
        "Symptoms": row["Symptoms"],  # Response from 'Symptoms' column
    }
    database.append(item)

# Precompute embeddings for each question-related field in batches
trigger_embeddings = embedding_model.encode(df["trigger_word"].tolist(), batch_size=32)
synonyms_embeddings = [
    embedding_model.encode(syn.split(","), batch_size=32) for syn in df["synonyms"]
]
keywords_embeddings = [
    embedding_model.encode(kw.split(","), batch_size=32) for kw in df["keywords"]
]

# Precompute embeddings for column names
column_names = ["What", "Why", "How", "Symptoms"]
column_embeddings = embedding_model.encode(column_names)

db_embeddings = []
for idx in range(len(df)):
    db_embeddings.append(
        {
            "trigger_embedding": trigger_embeddings[idx],
            "synonyms_embeddings": synonyms_embeddings[idx],
            "keywords_embeddings": keywords_embeddings[idx],
        }
    )

# Precompute embeddings for domain keywords
domain_keywords = ["heart", "cardiac", "women", "health", "cardiology"]
domain_embeddings = embedding_model.encode(domain_keywords)

best_match_response_flag = 0


def generate_response_with_placeholder(prompt):
    return "This is a placeholder response generated for your question."


def correct_spelling(text):
    if len(text.split()) > 1:
        corrected_words = [
            spellchecker.correction(word) if spellchecker.correction(word) else word
            for word in text.split()
        ]
        corrected_text = " ".join(corrected_words)
        return corrected_text
    return text


def lemmatize_query(query):
    return " ".join([lemmatizer.lemmatize(word) for word in query.split()])
    
#Calculate similarity scores for triggers, synonyms, and keywords.
def calculate_similarity(query_embedding, item_embeddings):
    trigger_score = cosine_similarity(query_embedding, item_embeddings['trigger_embedding'].reshape(1, -1)).flatten()[0]
    synonym_scores = [cosine_similarity(query_embedding, syn_emb.reshape(1, -1)).flatten()[0] for syn_emb in item_embeddings['synonyms_embeddings']]
    keyword_scores = [cosine_similarity(query_embedding, kw_emb.reshape(1, -1)).flatten()[0] for kw_emb in item_embeddings['keywords_embeddings']]
    return trigger_score, synonym_scores, keyword_scores

 #Determine if a strong direct match or an average score match is found.
def determine_match_type(trigger_score, max_synonym_score, max_keyword_score):
   
    if trigger_score >= 0.65 or max_synonym_score >= 0.65 or max_keyword_score >= 0.65:
        return "Max Match"
    return "Avg Max Match"

#Log the matching information.
def log_match_info(query, trigger_score, max_synonym_score, max_keyword_score, response, match_type):
    
    logging.info(
        f"{match_type} found. Query: '{query}', Trigger Score: {trigger_score:.4f}, "
        f"Synonym Score: {max_synonym_score:.4f}, Keyword Score: {max_keyword_score:.4f} "
        f"Response: {response}"
    )


def find_best_context(query, threshold):
    query_embedding = embedding_model.encode([query.lower()])
    
    best_score = 0
    best_response = None
    best_match_type = None
    
    max_score = 0
    max_response = None
    max_match_type = None
    
    for index, item_embeddings in enumerate(db_embeddings):
        trigger_score, synonym_scores, keyword_scores = calculate_similarity(query_embedding, item_embeddings)
        
        max_synonym_score = max(synonym_scores, default=0)
        max_keyword_score = max(keyword_scores, default=0)
        avg_score = (trigger_score + max_synonym_score + max_keyword_score ) / 3
        
        match_type = determine_match_type(trigger_score, max_synonym_score, max_keyword_score)
        
        if match_type == "Max Match":
            log_match_info(query, trigger_score, max_synonym_score, max_keyword_score, database[index], match_type)
            max_score = max(trigger_score, max_synonym_score, max_keyword_score)
            max_response = database[index]
            max_match_type = match_type
            
        if avg_score > best_score and all(score < 0.65 for score in [trigger_score, max_synonym_score, max_keyword_score]):
            log_match_info(query, trigger_score, max_synonym_score, max_keyword_score, database[index], match_type)
            best_score = avg_score
            best_response = database[index]
            best_match_type = match_type
    
    if best_score >= threshold and max_score < best_score:
        logging.info(
            f"Query: '{query}', Best Match Score: {best_score:.4f}, "
            f"Best Match Response: '{best_response}', Match Type: {best_match_type}"
        )

        return best_response
    
    elif max_score > best_score:

        logging.info(
            f"Query: '{query}', Max Match Score: {max_score:.4f}, "
            f"Best Match Response: '{max_response}', Match Type: {max_match_type}"
        )

        return max_response
    
    else:
        logging.warning(
            f"No suitable match found for query: '{query}' with score above threshold: {threshold}"
        )
        return None


def match_columns(query, best_match_response):

    query_lower =  correct_spelling(query).lower()

    intent_words = {
        "What": ["What","Define", "Identify","Describe","Clarify", "Specify","Detail", "Outline", "State", "Explain", "Determine","Depict", "Summarize","Designate", "Distinguish", ],
        "Symptoms": ["Symptoms","Signs","Indications","Manifestations", "Warning", "Clues","Evidence", "Redflags","Markers","Presentations", "Outcomes", "Patterns","Phenomena", "Traits","Occurrences", ],
        "Why": ["Why", "Causes", "Reason",  "Purpose", "Explain", "Justification", "Origin","Motive",  "Trigger", "Rationale","Grounds", "Basis", "Excuse", "Source", "Factor", ],
        "How": [ "How", "Method", "Means","Procedure", "Steps", "Technique", "Process", "Way", "Approach","Strategy", "System", "Manner", "Framework", "Form", "Mode", "Prevention",  "Avoidance","Safeguard", "Protection", "Mitigation","Reduction", "Intervention", "Defense", "Deterrence", "Shielding", ],

    }

    # Collect matching columns and their first occurrence positions
    matching_columns = []
    match_found = False  # Variable to track if a match is found

    #Check if the query contains any intent keywords
    for column, keywords in intent_words.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            position = query_lower.find(keyword_lower)
            if position != -1 and best_match_response.get(column):
                matching_columns.append((position, best_match_response[column]))
                match_found = True  # Set match_found to True when a match is found
                break  # Move to the next column once a match is found

    # If no match was found, add the response from the first column
    if not match_found and intent_words:
        first_column = next(iter(intent_words))  # Get the first column
        if best_match_response.get(first_column):
            default_response = best_match_response[first_column]
            best_match_response_flag = 1
            matching_columns.append((0, default_response))

    # Sort the matched columns by the position of their first occurrence in the query
    matching_columns.sort(key=lambda x: x[0])

    # Combine responses in the order of their appearance in the query
    responses = [response for _, response in matching_columns]

    # Return the combined response
    if responses:
        return " ".join(responses), best_match_response_flag

    # Fallback to the best matching column if no intent word is matched
    query_embedding = embedding_model.encode([query_lower])
    column_scores = cosine_similarity(query_embedding, column_embeddings).flatten()
    best_column_index = column_scores.argmax()
    best_column_name = column_names[best_column_index]
    logging.info( f"Best column match (fallback): {best_column_name} with score {column_scores[best_column_index]:.4f}" )

    return best_match_response.get(best_column_name, ""), best_match_response_flag


def is_domain_relevant(query, threshold=0.4):
    query_embedding = embedding_model.encode([query.lower()])
    relevance_scores = [
        cosine_similarity(query_embedding, [dom_emb]).flatten()[0]
        for dom_emb in domain_embeddings
    ]

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
            column_response, best_match_response_flag = match_columns(
                corrected_input, context_response
            )
            if best_match_response_flag == 1:
                column_response = (
                    column_response
                    + "\n For personalized advice or concerns about your health, Please consult our healthcare professional. We can provide you with the best guidance based on your specific needs."
                )
            return column_response

    logging.info(f"Checking Domain relevance")
    if is_domain_relevant(corrected_input):
        prompt = f"User asked: {corrected_input}. Please provide a helpful response related to women's heart health."
        logging.info(f"Prompt for Generative API: {prompt}")
        response = generate_response_with_placeholder(prompt)
        return response

    fallback_response = "I'm sorry, I can only answer questions related to women's heart health. Can you please clarify your question?"
    return fallback_response


logging.basicConfig(
    level=logging.INFO,
    filename="chatbot.log",
    filemode="a",
    format="%(asctime)s - %(message)s",
)


@app.route("/chatbot", methods=["POST"])
def chat():
    try:
        receieved_api_key = request.headers.get('X-API-KEY') 
        expected_api_key= 'fpv74NMceEzy.5OsNsX43uhfa2GSGPPOB1/o2ABXg0mMwniAef02'
        
        if receieved_api_key != expected_api_key:
            return jsonify({"unauthorized_access":"invalid api key"}), 401

        user_input = request.json.get("user_input", "").strip()

        if not user_input:
            return jsonify({"error": "Missing user input"}), 400

        response = get_response(user_input)

        return jsonify({"response": response}), 200

    except Exception as exception:
        logging.error(f"Error occurred: {str(exception)}")
        return jsonify({"error": str(exception)}), 500


if __name__ == "__main__":
    app.run(debug=True)
