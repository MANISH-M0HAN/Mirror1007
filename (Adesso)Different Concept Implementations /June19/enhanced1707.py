import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from spellchecker import SpellChecker
import numpy as np

class Chatbot:
    def __init__(self, df, embedding_model, tokenizer, model):
        self.df = df
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.model = model
        self.context_history = []
        self.spellchecker = SpellChecker()
        self.feedback_count = 0

    def correct_spelling(self, text):
        corrected_words = []
        for word in text.split():
            correction = self.spellchecker.correction(word)
            if correction and correction != word:
                corrected_words.append(correction)
            else:
                corrected_words.append(word)
        corrected_text = " ".join(corrected_words)
        print(f"Corrected Text: {corrected_text}")  # Debugging statement
        return corrected_text

    def find_best_context(self, query, threshold=0.4):  # Adjusted threshold
        query = self.correct_spelling(query)
        query_embedding = self.embedding_model.encode([query.lower()])
        embeddings = np.array(self.df['embedding'].tolist())
        similarities = cosine_similarity(query_embedding, embeddings).flatten()
        best_match_index = similarities.argmax()
        print(f"Query: {query}, Best Match Index: {best_match_index}, Similarity: {similarities[best_match_index]}")  # Debugging statement
        if similarities[best_match_index] < threshold:
            return None, None
        best_context = self.df.iloc[best_match_index]['response']
        return best_context, similarities[best_match_index]

    def generate_response(self, context):
        full_context = " ".join(self.context_history + [context])
        print(f"Full Context: {full_context}")  # Debugging statement
        inputs = self.tokenizer(full_context, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated Response: {response}")  # Debugging statement
        return response.strip()

    def get_response(self, user_input, threshold=0.4):  # Adjusted threshold
        user_input = self.correct_spelling(user_input)
        greetings = ["hello", "hi", "hey"]
        if any(greeting in user_input.lower() for greeting in greetings):
            return "Hello! How can I assist you with your heart health questions today?"

        context, similarity = self.find_best_context(user_input, threshold)
        if context is None:
            response = "I'm sorry, I don't understand. Can you please rephrase?"
        else:
            response = context
            self.context_history.append(user_input)
            self.context_history.append(response)

            if len(self.context_history) > 10:  # Limit context history to 10 exchanges
                self.context_history = self.context_history[-10:]

            self.feedback_count += 1
            if self.feedback_count >= 5:
                self.feedback_count = 0
                self.get_feedback()

        return response

    def get_feedback(self):
        feedback = input("Bot: Did you find this response helpful? (yes/no): ")
        if feedback.lower() in ["yes", "no"]:
            with open("feedback_log.txt", "a") as log:
                log.write(f"User input: {self.context_history[-2]}\nBot response: {self.context_history[-1]}\nFeedback: {feedback}\n\n")
        else:
            print("Bot: Invalid feedback. Please enter 'yes' or 'no'.")
            self.get_feedback()

def chatbot_interface():
    df = pd.read_csv('0856heart_health_triggers.csv')
    df['trigger_word'] = df['trigger_word'].str.lower()
    df['synonyms'] = df['synonyms'].str.lower()
    df['keywords'] = df['keywords'].str.lower()
    df['response'] = df['response'].astype(str).str.lower()
    
    # Create text_chunk for embeddings
    df['text_chunk'] = df[['trigger_word', 'synonyms', 'keywords', 'response']].apply(lambda x: ' '.join(x), axis=1)
    
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Improved model
    df['embedding'] = df['text_chunk'].apply(lambda x: embedding_model.encode(x))
    
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    model = AutoModelForCausalLM.from_pretrained('distilgpt2')
    
    chatbot = Chatbot(df, embedding_model, tokenizer, model)
    
    print("Welcome to the Women's Heart Health Chatbot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chatbot.get_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chatbot_interface()
