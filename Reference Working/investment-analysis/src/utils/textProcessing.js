// ./src/utils/textProcessing.js
const stopwords = require('stopword');
const natural = require('natural');

function preprocess(text) {
  const tokenizer = new natural.WordTokenizer();
  let tokens = tokenizer.tokenize(text);
  tokens = tokens.map(token => token.toLowerCase());
  tokens = stopwords.removeStopwords(tokens);
  return tokens.join(' ');
}

module.exports = preprocess;
