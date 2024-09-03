import nltk
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer

# Download necessary data for lemmatization
nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize the lemmatizer and spellchecker
lemmatizer = WordNetLemmatizer()
spellchecker = SpellChecker()

def correct_spelling(text):
    if len(text.split()) > 1:
        corrected_words = [
            spellchecker.correction(word) if spellchecker.correction(word) else word
            for word in text.split()
        ]
        corrected_text = " ".join(corrected_words)
        return corrected_text
    return text

def lemmatize_query(query):
    lemmatized_query = " ".join([lemmatizer.lemmatize(word) for word in query.split()])
    return lemmatized_query

