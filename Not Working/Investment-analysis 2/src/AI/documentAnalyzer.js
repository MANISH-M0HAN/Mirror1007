// ./src/AI/documentAnalyzer.js

const natural = require('natural');
const { pipeline } = require('transformers');
const preprocess = require('../utils/textProcessing');

class DocumentAnalyzer {
  constructor() {
    // Initialize sentiment analysis pipeline
    this.sentiment_analyzer = new natural.SentimentAnalyzer('English', natural.PorterStemmer, 'afinn');
    // Initialize BART model and tokenizer
    this.bart_model = pipeline('summarization');
  }

  summarizeText(text, max_chunk_length = 1024, max_summary_length = 200) {
    // Split the input text into chunks
    const chunks = [text.match(/.{1, max_chunk_length}/g)];
    const summaries = [];

    for (const chunk of chunks) {
      // Generate summary for each chunk
      const summary = this.bart_model(chunk, {
        max_length: max_summary_length,
        min_length: 30,
        length_penalty: 2.0,
        num_beams: 4,
        early_stopping: true
      });
      summaries.push(summary);
    }

    // Combine the summaries from all chunks
    const combined_summary = summaries.join(" ");
    return combined_summary;
  }

  analyzeSentiment(text) {
    const sentiment = this.sentiment_analyzer.getSentiment(preprocess(text));
    return sentiment;
  }

  performNER(text) {
    // You can use node-nlp or compromise for NER
    // Example:
    // const entities = new natural.NounInflector().pluralize(text);
    // return entities;
    return {};
  }

  analyzeDocument(text) {
    const summary = this.summarizeText(text);
    const sentiment_analysis = this.analyzeSentiment(text);
    const entities = this.performNER(text);

    const analysis = {
      summary,
      sentiment_analysis,
      named_entities: entities
    };

    return analysis;
  }
}

module.exports = DocumentAnalyzer;
