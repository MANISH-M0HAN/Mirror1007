// Function to add a message to the chat body
function addMessage(message, sender) {
    const chatBody = document.getElementById('chatbotBody');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', sender);
    messageElement.textContent = message;
    chatBody.appendChild(messageElement);
    chatBody.scrollTop = chatBody.scrollHeight;
}

// Function to handle sending a message
function sendMessage() {
    const userInput = document.getElementById('userInput').value;
    if (userInput) {
        addMessage(userInput, 'user');
        fetchResponse(userInput);
        document.getElementById('userInput').value = '';
    }
}

// Function to handle pre-typed messages
function preTypedMessage(message) {
    addMessage(message, 'user');
    fetchResponse(message);
}

// Function to fetch response from the server
async function fetchResponse(message) {
    const response = await fetch('http://localhost:3000/api/chatbot', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
    });

    const data = await response.json();
    addMessage(data.response, 'bot');
}

// Add initial welcome message
document.addEventListener('DOMContentLoaded', () => {
    addMessage('Hello! How can I help you today?', 'bot');
});
