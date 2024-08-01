# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from nltk.corpus import wordnet
# import nltk

# # Ensure NLTK WordNet is downloaded
# nltk.download('wordnet')

# # Load CSV file
# df = pd.read_csv('heart_health_triggers.csv')

# # Initialize the sentence transformer model
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Function to create text chunks (assuming each row is a chunk)
# def expand_synonyms(text):
#     synonyms = set()
#     for word in text.split():
#         for syn in wordnet.synsets(word):
#             for lemma in syn.lemmas():
#                 synonyms.add(lemma.name().lower())
#     return ' '.join(synonyms)

# # Apply synonym expansion and create text chunks
# df['text_chunk'] = df['trigger_word'] + ' ' + df['synonyms'] + ' ' + df['response']
# df['text_chunk'] = df['text_chunk'].apply(expand_synonyms)

# # Embedding all text chunks
# df['embedding'] = df['text_chunk'].apply(lambda x: embedding_model.encode(x))

# # Convert embeddings to list for FAISS
# embeddings = np.array(df['embedding'].tolist())
# embedding_matrix = np.vstack(embeddings)

# # Build the FAISS index
# dimension = embedding_matrix.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embedding_matrix)

# # Load a publicly available model
# tokenizer = AutoTokenizer.from_pretrained('microsoft/LLaMA')
# model = AutoModelForCausalLM.from_pretrained('microsoft/LLaMA')

# def generate_response(context):
#     inputs = tokenizer(context, return_tensors="pt", max_length=512, truncation=True)
#     outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response.strip()

# def get_response(user_input, embedding_model, index, df):
#     # Encode user input
#     user_embedding = embedding_model.encode([user_input])
    
#     # Perform semantic search
#     D, I = index.search(np.array(user_embedding), k=1)
    
#     print(f"User Input: {user_input}")
#     print(f"Semantic Search Results: Distance={D[0][0]}, Index={I[0][0]}")  # Debugging
    
#     if D[0][0] < 1.0:  # Adjusted threshold for semantic search
#         context = df.iloc[I[0][0]]['text_chunk']
#         print(f"Context found: {context}")  # Debugging
#         response = generate_response(context)
#         print(f"Generated Response: {response}")  # Debugging
#     else:
#         response = "I'm sorry, I don't understand. Can you please rephrase?"
    
#     return response

# def chatbot_interface():
#     print("Welcome to the Women's Heart Health Chatbot!")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             break
#         try:
#             response = get_response(user_input, embedding_model, index, df)
#         except Exception as e:
#             response = f"An error occurred: {str(e)}. Please try again later."
#         print(f"Bot: {response}")

# if __name__ == "__main__":
#     chatbot_interface()


# Kinda Okay but none still understand the test correctly.. i want it to be more better and improve.

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load CSV file
df = pd.read_csv('heart_health_triggers.csv')

# Initialize the sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to create text chunks (assuming each row is a chunk)
df['text_chunk'] = df['trigger_word'] + ' ' + df['synonyms'] + ' ' + df['response']
df['embedding'] = df['text_chunk'].apply(lambda x: embedding_model.encode(x))

# Convert embeddings to list for FAISS
embeddings = df['embedding'].tolist()
embedding_matrix = np.array(embeddings)

# Build the FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# Load a publicly available model
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

def generate_response(context):
    inputs = tokenizer(context, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def get_response(user_input, embedding_model, index, df):
    # Encode user input
    user_embedding = embedding_model.encode([user_input])
    
    # Perform semantic search
    D, I = index.search(np.array(user_embedding), k=1)
    
    if D[0][0] < 0.7:  # Define an appropriate threshold
        context = df.iloc[I[0][0]]['text_chunk']
        response = generate_response(context)
    else:
        response = "I'm sorry, I don't understand. Can you please rephrase?"
    
    return response

def chatbot_interface():
    print("Welcome to the Women's Heart Health Chatbot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = get_response(user_input, embedding_model, index, df)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chatbot_interface()



#-------------------Below is using distilgpt2 model which is set with higher threshhold to match the context, it works fine for 90% of texts---------------

# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load CSV file
# df = pd.read_csv('heart_health_triggers.csv')

# # Initialize the sentence transformer model
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Function to create text chunks (assuming each row is a chunk)
# df['text_chunk'] = df['trigger_word'] + ' ' + df['synonyms'] + ' ' + df['response']
# df['embedding'] = df['text_chunk'].apply(lambda x: embedding_model.encode(x))

# # Convert embeddings to list for FAISS
# embeddings = df['embedding'].tolist()
# embedding_matrix = np.array(embeddings)

# # Build the FAISS index
# dimension = embedding_matrix.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embedding_matrix)

# # Load a publicly available model
# tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
# model = AutoModelForCausalLM.from_pretrained('distilgpt2')

# def generate_response(context):
#     inputs = tokenizer(context, return_tensors="pt")
#     outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response.strip()

# def get_response(user_input, embedding_model, index, df):
#     # Encode user input
#     user_embedding = embedding_model.encode([user_input])
    
#     # Perform semantic search
#     D, I = index.search(np.array(user_embedding), k=1)
    
#     if D[0][0] < 0.7:  # Define an appropriate threshold
#         context = df.iloc[I[0][0]]['text_chunk']
#         response = generate_response(context)
#     else:
#         response = "I'm sorry, I don't understand. Can you please rephrase?"
    
#     return response

# def chatbot_interface():
#     print("Welcome to the Women's Heart Health Chatbot!")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             break
#         response = get_response(user_input, embedding_model, index, df)
#         print(f"Bot: {response}")

# if __name__ == "__main__":
#     chatbot_interface()



# #------------------------------------------Below is using huggling face Model which is better trained but cant use without authentication-------------------------------------



# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load CSV file
# df = pd.read_csv('heart_health_triggers.csv')

# # Initialize the sentence transformer model
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Function to create text chunks (assuming each row is a chunk)
# df['text_chunk'] = df['trigger_word'] + ' ' + df['synonyms'] + ' ' + df['response']
# df['embedding'] = df['text_chunk'].apply(lambda x: embedding_model.encode(x))

# # Convert embeddings to list for FAISS
# embeddings = df['embedding'].tolist()
# embedding_matrix = np.array(embeddings)

# # Build the FAISS index
# dimension = embedding_matrix.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(embedding_matrix)

# # Load Llama 2 model
# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b')
# model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b')

# def generate_response(context):
#     inputs = tokenizer(context, return_tensors="pt")
#     outputs = model.generate(inputs['input_ids'], max_length=100)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# def get_response(user_input, embedding_model, index, df):
#     # Encode user input
#     user_embedding = embedding_model.encode([user_input])
    
#     # Perform semantic search
#     D, I = index.search(np.array(user_embedding), k=1)
    
#     if D[0][0] < 0.7:  # Define an appropriate threshold
#         context = df.iloc[I[0][0]]['text_chunk']
#         response = generate_response(context)
#     else:
#         response = "I'm sorry, I don't understand. Can you please rephrase?"
    
#     return response

# def chatbot_interface():
#     print("Welcome to the Women's Heart Health Chatbot!")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             break
#         response = get_response(user_input, embedding_model, index, df)
#         print(f"Bot: {response}")

# if __name__ == "__main__":
#     chatbot_interface()
