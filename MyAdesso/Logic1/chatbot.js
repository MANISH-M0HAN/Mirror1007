// Function to toggle chat modal visibility
function toggleChat() {
    const modal = document.getElementById('chat-modal');
    modal.style.display = modal.style.display === 'none' ? 'block' : 'none';
}

// Function to send message to backend
async function sendMessage() {
    const inputField = document.getElementById('chat-input');
    const userMessage = inputField.value.trim();
    if (userMessage === '') return;

    // Append user message to chat
    appendMessage('user-message', userMessage);
    inputField.value = '';

    try {
        // Send request to backend
        const response = await fetch('http://127.0.0.1:5000/chatbot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user_input: userMessage }),
        });

        const data = await response.json();

        if (response.ok) {
            // Append bot response to chat
            appendMessage('bot-message', data.response);
        } else {
            appendMessage('bot-message', 'An error occurred. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        appendMessage('bot-message', 'Unable to reach the server. Please try again later.');
    }
}

// Function to append message to chat
function appendMessage(className, message) {
    const chatMessages = document.getElementById('chat-messages');
    const messageElement = document.createElement('div');
    messageElement.classList.add(className);
    messageElement.textContent = message;
    chatMessages.appendChild(messageElement);

    // Auto-scroll to the bottom of the chat messages
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Handle 'Enter' key press for sending message
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}
