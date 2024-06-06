const express = require('express');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

app.post('/api/chatbot', (req, res) => {
    const userMessage = req.body.message.toLowerCase();

    // Simple pre-typed responses
    let botResponse;
    if (userMessage.includes('hi') || userMessage.includes('hey')) {
        botResponse = 'Hello! How can I assist you today?';
    } else {
        botResponse = "I'm not sure how to respond to that. Can you try asking something else?";
    }

    res.json({ response: botResponse });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Chatbot server is running on port ${PORT}`);
});
