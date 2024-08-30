from spellchecker import SpellChecker

def correct_spelling(text):
    spellchecker= SpellChecker()
    if len(text.split()) > 1:
        corrected_words = [
            spellchecker.correction(word) if SpellChecker().correction(word) else word
            for word in text.split()
        ]
        corrected_text = ' '.join(corrected_words)
        return corrected_text
    return text
