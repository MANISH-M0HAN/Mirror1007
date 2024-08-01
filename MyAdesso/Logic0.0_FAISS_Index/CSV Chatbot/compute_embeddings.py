import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import faiss

# Load the data
data = pd.read_csv('responses.csv')

# Load pre-trained Huggingface model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def compute_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Compute embeddings for all responses
embeddings = compute_embeddings(data['response'].tolist())

# Set up FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the index and data
faiss.write_index(index, 'faiss_index.bin')
data.to_csv('responses_with_embeddings.csv', index=False)
