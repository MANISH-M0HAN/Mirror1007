# import requests

# # Replace 'your_huggingface_api_token' with your actual Hugging Face API token
# API_URL = "https://api-inference.huggingface.co/models/gpt2"
# headers = {"Authorization": "Bearer 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"}

# def query(payload):
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.json()

# # Define the system and user content
# system_content = "You are a travel agent. Be descriptive and helpful."
# user_content = "Tell me about San Francisco"

# # Combine system and user content to form the prompt
# prompt = f"{system_content}\nUser: {user_content}\nTravel Agent:"

# # Make a query to the API
# data = query({"inputs": prompt})

# # # Debug: Print the entire response to understand its structure
# # print("API Response:\n", data)

# # Extract and print the generated text with error handling
# try:
#     response_text = data[0]['generated_text']
#     print("Response:\n", response_text)
# except (KeyError, IndexError) as e:
#     print("Error extracting response text:", e)
#     print("Complete response data:", data)
import requests
import time

# Replace 'your_huggingface_api_token' with your actual Hugging Face API token
API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    while True:
        response = requests.post(API_URL, headers=headers, json=payload)
        data = response.json()
        if 'error' not in data:
            return data
        elif data['error'] == 'Model deepset/roberta-base-squad2 is currently loading':
            print("Model is loading. Waiting for it to finish...")
            time.sleep(40)  # Wait for 40 seconds before retrying
        else:
            return data  # Return error response if it's not related to model loading

# Define the user question
user_question = "Can you tell me something interesting about San Francisco?"

# Format the prompt with a mask token
prompt = f"User: {user_question}\nBot:"


# Make a query to the API
data = query({"inputs": prompt})

# Extract and print the generated text with error handling
try:
    if 'error' in data and 'No mask_token' in data['error']:
        print("Error: No mask token ([MASK]) found in the input. Please provide a prompt with a mask token.")
    else:
        response_text = data[0]['generated_text']
        print("Response:\n", response_text)
except (KeyError, IndexError) as e:
    print("Error extracting response text:", e)
    print("Complete response data:", data)
