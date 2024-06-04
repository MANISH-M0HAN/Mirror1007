import spacy

def extract_named_entities(text):
    """Extract named entities using spaCy."""
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        entities = {ent.label_: [ent.text] for ent in doc.ents}
        return entities
    except Exception as e:
        print("Error occurred during named entity recognition:", e)
        return {}
