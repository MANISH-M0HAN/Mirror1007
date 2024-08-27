# spell_checker.py

import csv
from difflib import get_close_matches
import re

# Common English words set
common_english_words = {
    "fault", "a", "ass", "i", "about", "above", "across", "after", "again", "against", "all", "almost", "along", 
    "think", "i am", "this", "amazing", "you", "are", "i think", "already", "also", "although", "always", 
    "am", "among", "an", "and", "another", "any", "anyone", "anything", "anywhere", "are", "around", "as", 
    "at", "away", "back", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each", "either", 
    "enough", "especially", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", 
    "few", "for", "from", "further", "get", "gets", "getting", "give", "go", "goes", "going", "gone", 
    "got", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", 
    "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "keep", 
    "keeps", "kind", "know", "knows", "knew", "known", "last", "later", "let", "lets", "like", "likely", 
    "look", "looking", "looks", "lot", "lots", "made", "make", "makes", "many", "may", "me", "might", 
    "mine", "more", "most", "much", "must", "my", "myself", "need", "needs", "never", "new", "next", 
    "no", "not", "nothing", "now", "of", "off", "often", "oh", "on", "once", "one", "only", "onto", 
    "or", "other", "others", "ought", "our", "ours", "ourselves", "out", "over", "own", "part", 
    "perhaps", "quite", "rather", "really", "right", "said", "same", "saw", "say", "says", "see", 
    "seem", "seemed", "seeming", "seems", "sees", "seen", "several", "shall", "she", "should", 
    "since", "so", "some", "somebody", "someone", "something", "sometimes", "somewhere", "still", 
    "such", "sure", "take", "takes", "taking", "tell", "than", "that", "the", "their", "theirs", 
    "them", "themselves", "then", "there", "therefore", "these", "they", "thing", "things", "think", 
    "thinks", "this", "those", "though", "thought", "thoughts", "through", "thus", "to", "together", 
    "too", "took", "toward", "under", "until", "up", "upon", "us", "use", "used", "uses", "using", 
    "very", "want", "wants", "was", "way", "we", "well", "went", "were", "what", "when", "where", 
    "whether", "which", "while", "who", "whom", "whose", "why", "will", "with", "within", "without", 
    "won't", "would", "yes", "yet", "you", "your", "yours", "yourself", "yourselves", "input", "read", "wake"
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
