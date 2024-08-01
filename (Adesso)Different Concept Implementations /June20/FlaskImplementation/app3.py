import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from spellchecker import SpellChecker
import numpy as np
from flask import Flask, request, jsonify, session
from flask_cors import CORS

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'randomtext'  # Replace with a random secret key for session management
CORS(app)

# Load data
df = pd.read_csv('heart_health_triggers.csv')
df.fillna('', inplace=True)  # Fill NaN values with empty strings
df['trigger_word'] = df['trigger_word'].str.lower()
df['synonyms'] = df['synonyms'].str.lower()
df['keywords'] = df['keywords'].str.lower()
df['response'] = df['response'].astype(str).str.lower()
df['text_chunk'] = df[['trigger_word', 'synonyms', 'keywords', 'response']].apply(lambda x: ' '.join(x), axis=1)

# Initialize models
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
df['embedding'] = df['text_chunk'].apply(lambda x: embedding_model.encode(x))
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
model = AutoModelForCausalLM.from_pretrained('distilgpt2')
spellchecker = SpellChecker()

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
    embeddings = np.array(df['embedding'].tolist())
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    best_match_index = similarities.argmax()
    if similarities[best_match_index] < threshold:
        return None, None
    best_context = df.iloc[best_match_index]['response']
    return best_context, similarities[best_match_index]

def generate_response(context, context_history):
    full_context = " ".join(context_history + [context])
    inputs = tokenizer(full_context, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def get_response(user_input, context_history, threshold=0.4):
    user_input = correct_spelling(user_input)
    greetings = ["hello", "hi", "hey"]

    if any(user_input.lower() == greeting for greeting in greetings):
        return "Hello! How can I assist you with your heart health questions today?", context_history

    context, similarity = find_best_context(user_input, threshold)
    if context is None:
        response = "I'm sorry, I don't understand. Can you please rephrase?"
    else:
        response = context
        context_history.append(user_input)
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
