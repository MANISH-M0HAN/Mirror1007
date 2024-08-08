// ./src/controllers/reportController.js

const documentAnalyzer = require('../AI/documentAnalyzer'); // Import the documentAnalyzer instance


exports.generateReport = (req, res) => {
  const responses = req.body.responses;

  if (!responses || responses.length === 0) {
    return res.status(400).send({ error: 'No survey responses provided.' });
  }

  const analysis = documentAnalyzer.analyzeDocument(responses); // Use the documentAnalyzer instance

  res.send({ analysis });
};
