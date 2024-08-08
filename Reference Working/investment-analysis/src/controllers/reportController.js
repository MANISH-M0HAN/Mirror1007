// ./src/controllers/reportController.js
const { calculateTfIdf, getTopKeywords } = require('../utils/tfidfHelper');

exports.generateReport = (req, res) => {
  const responses = req.body.responses;

  if (!responses || responses.length === 0) {
    return res.status(400).send({ error: 'No survey responses provided.' });
  }

  const tfidf = calculateTfIdf(responses);
  const summaries = getTopKeywords(tfidf, 5);  // Adjust the number of keywords as needed

  let report = "Investment Analysis Report\n\n";
  summaries.forEach((summary, index) => {
    report += `Summary ${index + 1}:\n`;
    report += `Top Keywords: ${summary}\n\n`;
  });

  const investmentWorthy = summaries.some(summary => summary.includes('profitable'));

  report += "Investment Worthiness: " + (investmentWorthy ? "Yes" : "No");

  res.send({ report });
};
