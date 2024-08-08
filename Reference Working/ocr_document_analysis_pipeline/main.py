import ssl
import nltk
import certifi
import os

# Fix SSL certificate error for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Suppress Hugging Face tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from ocr.ocr import extract_text_from_image
from text_preprocessing.preprocessing import preprocess_text
from document_analysis.summarization import summarize_text
from document_analysis.sentiment_analysis import analyze_sentiment
from document_analysis.named_entity_recognition import extract_named_entities
from document_analysis.topic_modeling import perform_topic_modeling

def main():
    # Specify the path to your image
    image_path = 'images/Sample_Image5.jpg'
    
    # Step 1: Extract text from the imagex
    extracted_text = extract_text_from_image(image_path)
    print("Extracted Text:", extracted_text)
    
    # Step 2: Preprocess the text
    preprocessed_text = preprocess_text(extracted_text)
    print("Preprocessed Text:", preprocessed_text)
    
    if len(preprocessed_text.split()) < 10:
        print("Text is too short for meaningful analysis.")
        return

    # Step 3: Summarize the text
    summary = summarize_text(preprocessed_text)
    print("Summary:", summary)
    
    # Step 4: Perform sentiment analysis
    sentiment = analyze_sentiment(preprocessed_text)
    print("Sentiment Analysis:", sentiment)
    
    # Step 5: Recognize named entities
    named_entities = extract_named_entities(preprocessed_text)
    print("Named Entities:", named_entities)
    
    # Step 6: Perform topic modeling (only if the text is long enough)
    if len(preprocessed_text.split()) > 20:  # Ensure enough words for topic modeling
        topics = perform_topic_modeling(preprocessed_text)
        print("Topics:", topics)
    else:
        print("Text is too short for topic modeling.")

if __name__ == "__main__":
    main()
