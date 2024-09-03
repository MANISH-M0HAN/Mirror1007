import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .config import Config

# Load the CSV file into a DataFrame
df = pd.read_csv(Config.CSV_FILE)
df.fillna("", inplace=True)

# Initialize the embedding model
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Create a database list from the DataFrame
database = []
for index, row in df.iterrows():
    item = {
        "trigger_word": row["trigger_word"],
        "synonyms": row["synonyms"].split(","),  # Assuming synonyms are comma-separated
        "keywords": row["keywords"].split(","),  # Assuming keywords are comma-separated
        "What": row["What"],  # Response from 'What' column
        "Why": row["Why"],  # Response from 'Why' column
        "How": row["How"],  # Response from 'How' column
        "Symptoms": row["Symptoms"],  # Response from 'Symptoms' column
    }
    database.append(item)

# Precompute embeddings for each question-related field
trigger_embeddings = embedding_model.encode(df["trigger_word"].tolist(), batch_size=32)
synonyms_embeddings = [embedding_model.encode(syn.split(","), batch_size=32) for syn in df["synonyms"]]
keywords_embeddings = [embedding_model.encode(kw.split(","), batch_size=32) for kw in df["keywords"]]
column_names = ["What", "Why", "How", "Symptoms"]
column_embeddings = embedding_model.encode(column_names)

db_embeddings = []
for idx in range(len(df)):
    db_embeddings.append(
        {
            "trigger_embedding": trigger_embeddings[idx],
            "synonyms_embeddings": synonyms_embeddings[idx],
            "keywords_embeddings": keywords_embeddings[idx],
        }
    )

domain_keywords = ["heart", "cardiac", "women", "health", "cardiology"]
domain_embeddings = embedding_model.encode(domain_keywords)

