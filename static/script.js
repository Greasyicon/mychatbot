// Initial mode set to 'chatbot'
let mode = 'chatbot';  

// Connection to the server using socket.io
const socket = io.connect('http://' + document.domain + ':' + location.port);

// Event listener for Chatbot Mode button
$('#chatbotMode').click(function() {
    mode = 'chatbot';
    $('#chatbotMode').addClass('btn-primary active').removeClass('btn-secondary');
    $('#docbotMode').addClass('btn-secondary').removeClass('btn-primary active');
});

// Event listener for Docbot Mode button
$('#docbotMode').click(function() {
    mode = 'docbot';
    $('#docbotMode').addClass('btn-primary active').removeClass('btn-secondary');
    $('#chatbotMode').addClass('btn-secondary').removeClass('btn-primary active');
});

// Event listener for form submission
$('#chatForm').submit(function(event) {
    event.preventDefault();  // Prevents the default form submission behavior
    const message = $('#userInput').val();
    socket.emit('message', { message: message, mode: mode });  // Emits the message to the server
    $('#userInput').val('');  // Clears the text area
});

// Event listener for Enter key press in textarea
$('#userInput').keypress(function(event) {
    if (event.which == 13 && !event.shiftKey) {  // Enter key pressed without shift
        event.preventDefault();  // Prevents new line
        $('#chatForm').submit();  // Triggers form submission
    }
});

// Event listener for successful connection to the server
socket.on('connect', () => {
    console.log('Connected to server');
});

// Event listener for receiving a message from the server
socket.on('message', (data) => {
    const now = new Date();  // Gets the current date and time
    const timestamp = now.toLocaleTimeString();  // Formats the time as a string
    // Determines the avatar image based on the user
    const avatar = data.user === 'You' ?
        '<img src="static/user_avatar.png" class="avatar" alt="User Avatar">' :
        '<img src="static/bot_avatar.png" class="avatar" alt="Bot Avatar">';
    // Appends a new message to the chat window
    $('#response').append(
        '<div class="message ' + (data.user === 'You' ? 'my-message' : 'maya-message') + '">' +
        avatar +
        '<strong>' + data.user + ':</strong> ' +
        data.text +
        '<div class="timestamp">' + timestamp + '</div>' +
        '</div>'
    );
    // Scrolls the card-body to the bottom to show the latest message
    $('.card-body').scrollTop($('.card-body')[0].scrollHeight);
    $('.chat-content').scrollTop($('.chat-content')[0].scrollHeight);

});

// This script ends here. Any other logic related to your chat interface would follow...
