import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

class Chatbot:
    def __init__(self, df, embedding_model, tokenizer, model):
        self.df = df
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.model = model
        self.context_history = []

    def find_best_context(self, query, threshold=0.5):
        query_embedding = self.embedding_model.encode([query.lower()])
        embeddings = self.df['embedding'].tolist()
        similarities = cosine_similarity(query_embedding, embeddings).flatten()
        best_match_index = similarities.argmax()
        if similarities[best_match_index] < threshold:
            return None, None
        best_context = self.df.iloc[best_match_index]['response']
        return best_context, similarities[best_match_index]

    def generate_response(self, context):
        full_context = " ".join(self.context_history + [context])
        inputs = self.tokenizer(full_context, return_tensors="pt")
        outputs = self.model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

    def get_response(self, user_input, threshold=0.5):
        context, _similarity = self.find_best_context(user_input, threshold)
        if context is None:
            response = "I'm sorry, I don't understand. Can you please rephrase?"
        else:
            response = self.generate_response(context)
        self.context_history.append(user_input)
        self.context_history.append(response)
        if len(self.context_history) > 10:  # Limit context history to 10 exchanges
            self.context_history = self.context_history[-10:]
        return response

    def get_feedback(self):
        feedback = input("Bot: Did you find this response helpful? (yes/no): ")
        if feedback.lower() in ["yes", "no"]:
            with open("feedback_log.txt", "a") as log:
                log.write(f"User input: {' '.join(self.context_history[-2])}\nBot response: {self.context_history[-1]}\nFeedback: {feedback}\n\n")
            return feedback.lower()
        else:
            print("Bot: Invalid feedback. Please enter 'yes' or 'no'.")
            return self.get_feedback()

def chatbot_interface():
    df = pd.read_csv('heart_health_triggers.csv')
    df['trigger_word'] = df['trigger_word'].str.lower()
    df['synonyms'] = df['synonyms'].str.lower()
    df['response'] = df['response'].astype(str).str.lower()
    df['text_chunk'] = df['trigger_word'] + ' ' + df['synonyms'] + ' ' + df['response']
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
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
        chatbot.get_feedback()

if __name__ == "__main__":
    chatbot_interface()
