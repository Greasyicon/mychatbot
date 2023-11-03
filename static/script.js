let mode = 'chatbot';  // default mode
const socket = io.connect('http://' + document.domain + ':' + location.port);

// Mode selector logic
$('#chatbotMode').click(function() {
    mode = 'chatbot';
    $('#chatbotMode').addClass('btn-primary active').removeClass('btn-secondary');
    $('#docbotMode').addClass('btn-secondary').removeClass('btn-primary active');
});

$('#docbotMode').click(function() {
    mode = 'docbot';
    $('#docbotMode').addClass('btn-primary active').removeClass('btn-secondary');
    $('#chatbotMode').addClass('btn-secondary').removeClass('btn-primary active');
});

// Form submission logic
$('#chatForm').submit(function(event) {
    event.preventDefault();  // Prevents the default form submission behavior
    const message = $('#userInput').val();
    socket.emit('message', { message: message, mode: mode });
    $('#userInput').val('');
});

// New logic to handle Enter key press in textarea
$('#userInput').keypress(function(event) {
    if (event.which == 13 && !event.shiftKey) {  // Enter key pressed without shift
        event.preventDefault();  // Prevents new line
        $('#chatForm').submit();  // Triggers form submission
    }
});

socket.on('connect', () => {
    console.log('Connected to server');
});

// Changed event name from 'response' to 'message'
// and adjusted the message display logic to match the updated server code
socket.on('message', (data) => {
    $('#response').append('<div class="message ' + (data.user === 'You' ? 'my-message' : 'maya-message') + '"><strong>' + data.user + ':</strong> ' + data.text + '</div>');
});
