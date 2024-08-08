function sendMessage() {
    var inputElement = document.getElementById('user-input');
    var message = inputElement.value.trim();
    if (message) {
        displayMessage('User', message);
        // Send message to the backend API
        fetch('http://127.0.0.1:5000/get-response', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        })
            .then(function (response) { return response.json(); })
            .then(function (data) {
            displayMessage('Bot', data.response);
        })
            .catch(function (error) {
            console.error('Error:', error);
            displayMessage('Bot', 'Sorry, something went wrong.');
        });
        inputElement.value = '';
    }
}
function sendButtonMessage(keyword) {
    displayMessage('User', keyword);
    // Fetch response for the keyword
    fetch('http://127.0.0.1:5000/get-response', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: keyword }),
    })
        .then(function (response) { return response.json(); })
        .then(function (data) {
        displayMessage('Bot', data.response);
    })
        .catch(function (error) {
        console.error('Error:', error);
        displayMessage('Bot', 'Sorry, something went wrong.');
    });
}
function displayMessage(sender, message) {
    var chatbox = document.getElementById('chatbox');
    var messageElement = document.createElement('p');
    messageElement.innerHTML = "<strong>".concat(sender, ":</strong> ").concat(message);
    chatbox.appendChild(messageElement);
    chatbox.scrollTop = chatbox.scrollHeight; // Scroll to the bottom
}
// Attach event listener to the send button
var sendButton = document.querySelector('button[onclick="sendMessage()"]');
sendButton === null || sendButton === void 0 ? void 0 : sendButton.addEventListener('click', sendMessage);
// Attach event listeners to predefined query buttons
var symptomButton = document.querySelector('button[onclick="sendButtonMessage(\'symptoms\')"]');
symptomButton === null || symptomButton === void 0 ? void 0 : symptomButton.addEventListener('click', function () { return sendButtonMessage('symptoms'); });
var preventionButton = document.querySelector('button[onclick="sendButtonMessage(\'prevention\')"]');
preventionButton === null || preventionButton === void 0 ? void 0 : preventionButton.addEventListener('click', function () { return sendButtonMessage('prevention'); });
var treatmentButton = document.querySelector('button[onclick="sendButtonMessage(\'treatment\')"]');
treatmentButton === null || treatmentButton === void 0 ? void 0 : treatmentButton.addEventListener('click', function () { return sendButtonMessage('treatment'); });
