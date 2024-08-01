function sendMessage(): void {
    const inputElement = document.getElementById('user-input') as HTMLInputElement;
    const message = inputElement.value.trim();

    if (message) {
        displayMessage('User', message);

        // Send message to the backend API
        fetch('http://127.0.0.1:5000/get-response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message }),
        })
        .then(response => response.json())
        .then(data => {
            displayMessage('Bot', data.response);
        })
        .catch(error => {
            console.error('Error:', error);
            displayMessage('Bot', 'Sorry, something went wrong.');
        });

        inputElement.value = '';
    }
}

function sendButtonMessage(keyword: string): void {
    displayMessage('User', keyword);

    // Fetch response for the keyword
    fetch('http://127.0.0.1:5000/get-response', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: keyword }),
    })
    .then(response => response.json())
    .then(data => {
        displayMessage('Bot', data.response);
    })
    .catch(error => {
        console.error('Error:', error);
        displayMessage('Bot', 'Sorry, something went wrong.');
    });
}

function displayMessage(sender: string, message: string): void {
    const chatbox = document.getElementById('chatbox') as HTMLDivElement;
    const messageElement = document.createElement('p');
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatbox.appendChild(messageElement);
    chatbox.scrollTop = chatbox.scrollHeight; // Scroll to the bottom
}

// Attach event listener to the send button
const sendButton = document.querySelector('button[onclick="sendMessage()"]');
sendButton?.addEventListener('click', sendMessage);

// Attach event listeners to predefined query buttons
const symptomButton = document.querySelector('button[onclick="sendButtonMessage(\'symptoms\')"]');
symptomButton?.addEventListener('click', () => sendButtonMessage('symptoms'));

const preventionButton = document.querySelector('button[onclick="sendButtonMessage(\'prevention\')"]');
preventionButton?.addEventListener('click', () => sendButtonMessage('prevention'));

const treatmentButton = document.querySelector('button[onclick="sendButtonMessage(\'treatment\')"]');
treatmentButton?.addEventListener('click', () => sendButtonMessage('treatment'));
