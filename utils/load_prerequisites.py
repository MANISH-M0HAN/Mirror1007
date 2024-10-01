import os
from sentence_transformers import SentenceTransformer
import pandas as pd
from dotenv import load_dotenv

#Initialize models and spellchecker
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the CSV file into a DataFrame
csv_file = 'heart_health_triggers.csv'
df = pd.read_csv(csv_file)
df.fillna('Nan', inplace=True)

# Create a database list from the DataFrame
database = []
for index, row in df.iterrows():
    item = {
        "trigger_words": row['trigger_words'].split(','),
        "synonyms": row['synonyms'].split(','),
        "keywords": row['keywords'].split(','),
        "What": row['What'],
        "Why": row['Why'],
        "How": row['How'],
        "Symptoms": row['Symptoms']
    }
    database.append(item)

# Precompute embeddings for each question-related field in batches 
triggers_embeddings = [embedding_model.encode(trw.split(','), batch_size=32) for trw in df['trigger_words']]
synonyms_embeddings = [embedding_model.encode(syn.split(','), batch_size=32) for syn in df['synonyms']]
keywords_embeddings = [embedding_model.encode(kw.split(','), batch_size=32) for kw in df['keywords']]

db_embeddings = []
for idx in range(len(df)):
    db_embeddings.append({
        "triggers_embeddings": triggers_embeddings[idx],
        "synonyms_embeddings": synonyms_embeddings[idx],
        "keywords_embeddings": keywords_embeddings[idx]
    })

# Populate this list with words that can sign that chat-agent when it does not find any answer
# Triggers the AI Method with the user input and set prompt
domain_keywords = ['women', 'health','MyAdesso']

# Below lines were added to create a domain list from the CSV data, 
# but it is redundant after using trigger_words, synonyms and keywords for direct match

# for index, item_embeddings in enumerate(db_embeddings):
#         trigger_words = [trigger.lower().strip() for trigger in database[index]["trigger_words"]]
#         synonyms = [synonym.lower().strip() for synonym in database[index]["synonyms"]]
#         keywords = [keyword.lower().strip() for keyword in database[index]["keywords"]]
#         domain_keywords.extend(trigger_words + synonyms + keywords)
# domain_keywords = list(set(domain_keywords))
domain_embeddings = embedding_model.encode(domain_keywords)