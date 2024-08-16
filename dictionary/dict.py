import csv
from difflib import get_close_matches
import re
import nltk
from nltk.corpus import words

# Ensure that the word list is downloaded
nltk.download('words')

# Load the common English words using nltk
common_english_words = set(words.words())

# Load Words from Multiple Columns
def load_word_set(csv_file, column_names):
    word_set = common_english_words  # Use the nltk word set
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for col in column_names:
                # Split by either semicolon or comma using regex
                phrases = re.split(r'[;,]', row[col].lower())
                for phrase in phrases:
                    # Further split each phrase by spaces if needed
                    words = phrase.split()
                    word_set.add(phrase.strip())
                    for word in words:
                        word_set.add(word.strip())  # Add words to the set
    return word_set

# Function to Correct Spellings Using Fuzzy Matching
def correct_spelling(text, word_set, cutoff=0.8):
    words = text.split()
    corrected_words = []
    i = 0

    while i < len(words):
        word = words[i].lower()
        match_found = False
        
        # Check combinations of up to 3 words
        for j in range(3, 0, -1):  # Start with 3-word combinations down to 1-word
            combined_word = ' '.join(words[i:i+j]).lower()
            matches = get_close_matches(combined_word, word_set, n=1, cutoff=cutoff)
            if matches:
                corrected_words.append(matches[0])
                i += j  # Skip the matched words
                match_found = True
                break
        
        if not match_found:
            corrected_words.append(words[i])
            i += 1

    return ' '.join(corrected_words)

# Example Usage:
if __name__ == "__main__":
    # Load the word set from multiple columns
    word_set = load_word_set('heart_health_triggers.csv', ['trigger_word','synonyms','keywords','category','sub_category','response'])
    
    print("Enter your text (type 'exit' to quit):")
    
    while True:
        # Get input from the user
        text = input(">> ")
        # Correct the text using fuzzy matching
        corrected_text = correct_spelling(text, word_set)
        print("Corrected Text:", corrected_text)

        # Check if the user wants to exit
        if text.lower() == 'exit':
            print("Exiting the program.")
            break

