from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import os

# Set the environment variable to allow duplicate OpenMP libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = Flask(__name__)
CORS(app)

# Load the pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load the data and FAISS index
data = pd.read_csv('responses_with_embeddings.csv')
index = faiss.read_index('faiss_index.bin')

def compute_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).numpy()
    return embedding

@app.route('/get-response', methods=['POST'])
def get_response():
    user_message = request.json['message']
    embedding = compute_embedding(user_message)
    _, indices = index.search(embedding, 1)
    response_index = indices[0][0]
    response = data['response'].iloc[response_index]
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
