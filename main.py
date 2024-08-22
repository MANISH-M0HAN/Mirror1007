# main.py

from spell_checker import load_word_set, correct_spelling

# Example Usage
if __name__ == "__main__":
    # Load the word set from multiple columns
    word_set = load_word_set('heart_health_triggers.csv', 
                             ['trigger_word', 'synonyms', 'keywords', 
                              'category', 'sub_category', 'response'])

    print("Enter your text (type 'exit' to quit):")
    while True:
        text = input(">> ")
        corrected_text = correct_spelling(text, word_set)
        print("Corrected Text:", corrected_text)

        if text.lower() == 'exit':
            print("Exiting the program.")
            break
