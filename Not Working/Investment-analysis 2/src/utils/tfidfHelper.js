// ./src/utils/tfidfHelper.js

const natural = require('natural');
const preprocess = require('./textProcessing');
const { TfIdf } = natural;

function calculateTfIdf(responses) {
  const tfidf = new TfIdf();
  responses.forEach(response => {
    tfidf.addDocument(preprocess(response));
  });
  return tfidf;
}

function getTopKeywords(tfidf, numKeywords) {
  const summaries = [];
  for (let i = 0; i < tfidf.documents.length; i++) {
    const items = tfidf.listTerms(i).slice(0, numKeywords);
    summaries.push(items.map(item => item.term).join(', '));
  }
  return summaries;
}

module.exports = { calculateTfIdf, getTopKeywords };
