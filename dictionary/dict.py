print("hello")
import csv
from difflib import get_close_matches
import re


'''common_english_words = {
    "has", "on", "in", "the", "and", "is", "of", "to", "with", "a", "an", "it", "for", "by", "this",
    "that", "from", "as", "at", "be", "but", "or", "are", "was", "were", "will", "would", "can", 
    "could", "should", "may", "might", "must", "have", "has", "had", "do", "does", "did", "am", "fault"
}'''

common_english_words = {
    "fault", "a", "i", "about", "above", "across", "after", "again", "against", "all", "almost", "along","think", "i am", "this", "amazing", "you", "are" , "i think",
    "already", "also", "although", "always", "am", "among", "an", "and", "another", "any",
    "anyone", "anything", "anywhere", "are", "around", "as", "at", "away", "back", "be",
    "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "cannot", "could", "did", "do", "does", "doing", "down", "during", "each", "either",
    "enough", "especially", "etc", "even", "ever", "every", "everyone", "everything", "everywhere",
    "few", "for", "from", "further", "get", "gets", "getting", "give", "go", "goes",
    "going", "gone", "got", "had", "has", "have", "having", "he", "her", "here", 
    "hers", "herself", "him", "himself", "his", "how", "however", "i", "if", "in", 
    "into", "is", "it", "its", "itself", "just", "keep", "keeps", "kind", "know",
    "knows", "knew", "known", "last", "later", "let", "lets", "like", "likely", "look",
    "looking", "looks", "lot", "lots", "made", "make", "makes", "many", "may", "me",
    "might", "mine", "more", "most", "much", "must", "my", "myself", "need", "needs",
    "never", "new", "next", "no", "not", "nothing", "now", "of", "off", "often",
    "oh", "on", "once", "one", "only", "onto", "or", "other", "others", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "part", "perhaps", "quite", "rather",
    "really", "right", "said", "same", "saw", "say", "says", "see", "seem", "seemed",
    "seeming", "seems", "sees", "seen", "several", "shall", "she", "should", "since", "so",
    "some", "somebody", "someone", "something", "sometimes", "somewhere", "still", "such", "sure", "take",
    "takes", "taking", "tell", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "therefore", "these", "they", "thing", "things", "think", "thinks", "this",
    "those", "though", "thought", "thoughts", "through", "thus", "to", "together", "too", "took",
    "toward", "under", "until", "up", "upon", "us", "use", "used", "uses", "using",
    "very", "want", "wants", "was", "way", "we", "well", "went", "were", "what",
    "when", "where", "whether", "which", "while", "who", "whom", "whose", "why", "will",
    "with", "within", "without", "won't", "would", "yes", "yet", "you", "your", "yours",
    "yourself", "yourselves", "able", "above", "accept", "according", "across", "act", "actually", "add",
    "after", "again", "against", "ago", "ahead", "allow", "almost", "alone", "along", "already",
    "although", "always", "among", "amount", "an", "and", "another", "answer", "any", "anybody",
    "anymore", "anyone", "anything", "anyway", "anywhere", "appear", "are", "area", "aren't", "around",
    "as", "ask", "asked", "asking", "asks", "at", "available", "away", "back", "bad",
    "be", "became", "because", "become", "becomes", "been", "before", "began", "begin", "begins",
    "behind", "being", "believe", "best", "better", "between", "big", "bit", "both", "bring",
    "but", "by", "call", "came", "can", "can't", "cannot", "care", "case", "cause",
    "certain", "certainly", "chance", "change", "clear", "clearly", "close", "come", "comes", "company",
    "complete", "consider", "continue", "could", "course", "create", "current", "decide", "decided", "decision",
    "definitely", "despite", "did", "didn't", "different", "directly", "does", "doesn't", "doing", "done",
    "don't", "down", "during", "each", "early", "either", "else", "end", "enough", "even",
    "ever", "every", "everybody", "everyone", "everything", "exactly", "expect", "experience", "explain", "far",
    "felt", "few", "finally", "find", "fine", "first", "follow", "following", "for", "found",
    "four", "from", "full", "gave", "get", "gets", "getting", "give", "given", "gives",
    "go", "going", "gone", "good", "got", "great", "had", "happen", "happened", "happens",
    "hard", "has", "have", "having", "he", "head", "hear", "heard", "help", "her",
    "here", "herself", "high", "him", "himself", "his", "hold", "home", "hope", "how",
    "however", "I", "I'd", "I'll", "I'm", "I've", "if", "important", "in", "inside",
    "instead", "into", "is", "isn't", "it", "it's", "its", "itself", "just", "keep",
    "kept", "kind", "knew", "know", "known", "large", "last", "later", "lead", "least",
    "leave", "left", "less", "let", "let's", "likely", "like", "long", "look", "looked",
    "looking", "looks", "lot", "made", "main", "make", "makes", "making", "man", "many",
    "matter", "may", "maybe", "mean", "means", "meant", "meet", "might", "mind", "minute",
    "moment", "more", "most", "mostly", "move", "much", "must", "my", "myself", "name",
    "need", "needed", "needs", "never", "new", "next", "no", "nobody", "non", "none",
    "not", "nothing", "now", "number", "of", "off", "often", "oh", "okay", "old",
    "on", "once", "one", "only", "onto", "or", "order", "other", "others", "our",
    "out", "over", "own", "part", "past", "people", "perhaps", "place", "plan", "point",
    "possible", "probably", "put", "question", "quickly", "quite", "rather", "reach", "ready", "real",
    "really", "reason", "received", "recently", "right", "said", "same", "saw", "say", "says",
    "second", "see", "seem", "seemed", "seems", "sell", "send", "sent", "set", "several",
    "she", "should", "show", "showed", "side", "simple", "simply", "since", "small", "so",
    "some", "somebody", "someone", "something", "sometimes", "soon", "sorry", "sort", "still", "such",
    "sure", "take", "taken", "takes", "taking", "tell", "ten", "than", "thank", "thanks",
    "that", "that's", "the", "their", "them", "themselves", "then", "there"}

# Load Words from Multiple Columns
def load_word_set(csv_file, column_names):
    word_set = set(common_english_words)  # Use a set to avoid duplicates
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
    # Example text
    #text = "This is an analisis of speeling and recieveing data."
    
    # Correct the text using fuzzy matching
    #corrected_text = correct_spelling(text, word_set)
    #print("Corrected Text:", corrected_text)

