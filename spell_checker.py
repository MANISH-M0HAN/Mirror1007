# spell_checker.py

import csv
from difflib import get_close_matches
import re

# Common English words set
common_english_words = {
    # ... (your common words set here)
}

# Load Words from Multiple Columns
def load_word_set(csv_file, column_names):
    word_set = set(common_english_words)  # Use a set to avoid duplicates
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for col in column_names:
                if col in ['trigger_word', 'synonyms', 'keywords', 'category', 'sub_category', 'response']:
                    phrases = re.split(r'[;,]', row[col].lower())  # Split by semicolon or comma
                    for phrase in phrases:
                        words = phrase.split()  # Further split each phrase by spaces
                        word_set.add(phrase.strip())  # Add full phrase to the set
                        for word in words:
                            word_set.add(word.strip())  # Add individual words to the set
    return word_set

# Function to Correct Spellings Using Fuzzy Matching
def correct_spelling(text, word_set, cutoff=0.85):
    words = text.split()
    corrected_words = []
    i = 0

    while i < len(words):
        best_match = None
        best_match_score = 0
        
        # Check combinations of up to 3 words
        for j in range(2, 0, -1):  # Start with 3-word combinations down to 1-word
            combined_word = ' '.join(words[i:i+j]).lower()
            matches = get_close_matches(combined_word, word_set, n=3, cutoff=cutoff)
            print ("matches :",matches) #display the possible matches
            
            # Find the best match from the available matches
            if matches:
                match = matches[0]
                match_score = len(match) / len(combined_word)  # Example scoring based on length match
                if match_score > best_match_score:
                    best_match = match
                    best_match_score = match_score
        
        if best_match:
            corrected_words.append(best_match)
            i += best_match.count(' ') + 1  # Skip the matched words
        else:
            corrected_words.append(words[i])
            i += 1

    return ' '.join(corrected_words)
