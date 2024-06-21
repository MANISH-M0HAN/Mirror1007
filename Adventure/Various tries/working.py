# # import pandas as pd
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise import cosine_similarity
# # from transformers import AutoModelForCausalLM, AutoTokenizer

# # # Load and preprocess data
# # df = pd.read_csv('heart_health_triggers.csv')
# # df['trigger_word'] = df['trigger_word'].str.lower()
# # df['synonyms'] = df['synonyms'].str.lower()
# # df['response'] = df['response'].astype(str).str.lower()
# # df['text_chunk'] = df['trigger_word'] + ' ' + df['synonyms']

# # # Combine all text chunks for vectorization
# # all_texts = df['text_chunk'].tolist()

# # # Initialize and fit TF-IDF Vectorizer
# # vectorizer = TfidfVectorizer().fit(all_texts)
# # text_vectors = vectorizer.transform(all_texts)

# # # Load a publicly available model
# # tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
# # model = AutoModelForCausalLM.from_pretrained('distilgpt2')

# # def find_best_response(query, vectorizer, text_vectors, df):
# #     query_vector = vectorizer.transform([query.lower()])
# #     similarities = cosine_similarity(query_vector, text_vectors).flatten()
# #     best_match_index = similarities.argmax()
# #     best_response = df.iloc[best_match_index]['response']
# #     return best_response

# # def generate_response(context):
# #     inputs = tokenizer(context, return_tensors="pt")
# #     outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
# #     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# #     return response.strip()

# # def get_response(user_input, vectorizer, text_vectors, df):
# #     context = find_best_response(user_input, vectorizer, text_vectors, df)
# #     response = generate_response(context)
# #     return response

# # def chatbot_interface():
# #     print("Welcome to the Women's Heart Health Chatbot!")
# #     while True:
# #         user_input = input("You: ")
# #         if user_input.lower() == "exit":
# #             break
# #         response = get_response(user_input, vectorizer, text_vectors, df)
# #         print(f"Bot: {response}")

# # if __name__ == "__main__":
# #     chatbot_interface()

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load and preprocess data
# df = pd.read_csv('heart_health_triggers.csv')
# df['trigger_word'] = df['trigger_word'].str.lower()
# df['synonyms'] = df['synonyms'].str.lower()
# df['response'] = df['response'].astype(str).str.lower()
# df['text_chunk'] = df['trigger_word'] + ' ' + df['synonyms'] + ' ' + df['response']

# # Combine all text chunks for vectorization
# all_texts = df['text_chunk'].tolist()

# # Initialize and fit TF-IDF Vectorizer
# vectorizer = TfidfVectorizer().fit(all_texts)
# text_vectors = vectorizer.transform(all_texts)

# # Load a publicly available model
# tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
# model = AutoModelForCausalLM.from_pretrained('distilgpt2')

# def find_best_context(query, vectorizer, text_vectors, df, threshold=0.2):
#     query_vector = vectorizer.transform([query.lower()])
#     similarities = cosine_similarity(query_vector, text_vectors).flatten()
#     best_match_index = similarities.argmax()
#     if similarities[best_match_index] < threshold:
#         return None, None
#     best_context = df.iloc[best_match_index]['response']
#     return best_context, similarities[best_match_index]

# def generate_response(context):
#     inputs = tokenizer(context, return_tensors="pt")
#     outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response.strip()

# def get_response(user_input, vectorizer, text_vectors, df, threshold=0.2):
#     context, similarity = find_best_context(user_input, vectorizer, text_vectors, df, threshold)
#     if context is None:
#         return "I'm sorry, I don't understand. Can you please rephrase?"
#     response = generate_response(context)
#     return response

# def chatbot_interface():
#     print("Welcome to the Women's Heart Health Chatbot!")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             break
#         response = get_response(user_input, vectorizer, text_vectors, df)
#         print(f"Bot: {response}")

# if __name__ == "__main__":
#     chatbot_interface()
