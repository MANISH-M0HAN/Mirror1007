import os
import google.generativeai as genai
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker
import pandas as pd
from dotenv import load_dotenv
import logging

# Set the TOKENIZERS_PARALLELISM environment variable to avoid deadlock warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
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
    corrected_words = [
        spellchecker.correction(word) if spellchecker.correction(word) else word
        for word in text.split()
    ]
    corrected_text = ' '.join(corrected_words)
    return corrected_text

def find_best_context(query, threshold=0.4):
    """
    This function takes a query and returns the best matching context from the database based on cosine similarity.
    If no match is found above the threshold, it returns None.
    """
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

def generate_response_with_gemini(prompt, original_response):
    """
    This function uses the Gemini API to generate a response for a given prompt.
    It catches exceptions and returns an error message if the API call fails.
    """
    try:
        response = gemini_model.generate_content(prompt)
        if response and hasattr(response, 'text'):
            response = response.text.strip()
            
            # Ensure the response doesn't deviate significantly from the original
            if len(response) > len(original_response) * 1.5 or '🤖' in response:  # Example condition to avoid unwanted AI-generated styles
                return original_response  # Fallback to original if too different
        else:
            response = original_response  # Use original if Gemini fails to produce a coherent response

        return response
    except Exception as e:
        logging.error(f"Error generating response with Gemini: {e}")
        return original_response  # Fallback to original on error

def get_relevant_context(user_input, context_history):
    """
    This function retrieves relevant context from the context history based on the user input.
    It helps in providing a coherent conversation by maintaining context.
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
    This function handles the logic to decide whether to use a pre-defined response or generate one with the Gemini API.
    It returns a response and updates the context history.
    """
    user_input = correct_spelling(user_input)
    
    greetings = ["hello", "hi", "hey"]

    if any(user_input.lower() == greeting for greeting in greetings):
        return "Hello! How can I assist you with your heart health questions today?", context_history

    # Check if a survey is active
    if 'survey' in session and session['survey'].get('active'):
        response = handle_survey(user_input)
        context_history['history'].append({"user_input": user_input, "bot_response": response})
        session['context_history'] = context_history
        return response, context_history

    context_response = find_best_context(user_input, threshold)
    if context_response:
        context_history['context'].append({"info": context_response})
        context_history['history'].append({"user_input": user_input, "bot_response": ""})

        relevant_context = get_relevant_context(user_input, context_history)
        prompt = (f"User asked: {user_input}\nContext: {relevant_context}\nMatched response: {context_response}\n"
                  "Please enhance slightly, making the text more coherent and user-friendly. "
                  "Avoid using any emojis or AI disclaimers, and ensure the response stays very close to the provided context.")
        
        logging.info(f"Prompt for Gemini API: {prompt}")
        
        # Send request to Gemini API with controlled enhancement
        response = generate_response_with_gemini(prompt, context_response)

        # Update context history with the response from Gemini
        context_history['history'][-1]['bot_response'] = response
        if len(context_history['history']) > 10:  # Limit context history to 10 exchanges
            context_history['history'] = context_history['history'][-10:]

        return response, context_history

    # Fallback response when no match is found in the CSV
    fallback_response = "I'm sorry, I can't help with that question. Please ask something related to heart health."
    context_history['history'].append({"user_input": user_input, "bot_response": fallback_response})
    return fallback_response, context_history

# Define the survey questions
survey_questions = [
    "What is your age?",
    "What is your gender?",
    "How would you rate your heart health on a scale of 1 to 10?",
    "Do you have any history of heart disease in your family?",
    "How often do you exercise in a week?"
]

def start_survey():
    """
    Initializes a new survey session and returns the first question.
    """
    session['survey'] = {
        'active': True,
        'current_question_index': 0,
        'responses': []
    }
    return survey_questions[0]  # Ask the first question

def handle_survey(user_input):
    """
    Handles user input during an active survey session.
    """
    survey_state = session['survey']
    current_index = survey_state['current_question_index']
    
    # Store the user's response
    survey_state['responses'].append(user_input)
    
    # Check if there are more questions
    if current_index + 1 < len(survey_questions):
        survey_state['current_question_index'] += 1
        session['survey'] = survey_state
        next_question = survey_questions[survey_state['current_question_index']]
        return f"Thank you! Now, {next_question}"
    else:
        # Survey is complete
        survey_state['active'] = False
        responses = survey_state['responses']
        session['survey'] = survey_state
        logging.info(f"Survey completed with responses: {responses}")
        return "Thank you for completing the survey! Your responses have been recorded."

@app.route('/start-survey', methods=['POST'])
def start_survey_route():
    """
    Endpoint to initiate a survey.
    """
    if 'survey' in session and session['survey'].get('active'):
        return jsonify({"response": "A survey is already in progress."})
    
    first_question = start_survey()
    return jsonify({"response": f"We have a quick survey for you! {first_question}"})

# Setup logging
logging.basicConfig(level=logging.INFO, filename='chatbot.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """
    This is the main endpoint for handling user input and returning the chatbot response.
    It uses session to maintain context history.
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid input"}), 400
        
        data = request.get_json()
        user_input = data.get('user_input')
        if 'context_history' not in session:
            session['context_history'] = {"context": [], "history": []}
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
