import csv

def load_responses(csv_file):
    responses = {}
    with open(csv_file, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            trigger_word = row['trigger_word'].strip().lower()
            responses[trigger_word] = row['response']
            synonyms = row.get('synonyms', '').strip()
            if synonyms:
                for synonym in synonyms.split(','):
                    responses[synonym.strip().lower()] = row['response']
    return responses

def find_response(user_query, responses):
    query_tokens = user_query.lower().split()
    for token in query_tokens:
        if token in responses:
            return responses[token]
    return "I'm sorry, I don't have information on that specific topic."

# Example usage:
csv_file = 'heart_health_triggers.csv'  # Replace with your CSV file path
responses = load_responses(csv_file)

user_query = "I am feeling sick today and have chest pain"
response = find_response(user_query, responses)
print(response)
