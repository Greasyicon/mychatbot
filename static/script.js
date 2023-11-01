let mode = 'chatbot';  // default mode
const socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('response', (msg) => {
    $('#response').append('<p><strong>Maya:</strong> ' + msg.response + '</p>');
});

socket.on('message', (data) => {
    $('#response').append('<div class="message ' + (data.user === 'You' ? 'my-message' : 'maya-message') + '"><strong>' + data.user + ':</strong> ' + data.text + '</div>');
});

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
$('#submit').click(() => {
    const message = $('#userInput').val();
    socket.emit('send-message', { message: message, mode: mode });
    $('#response').append('<p><strong>You:</strong> ' + message + '</p>');
    $('#userInput').val('');
});
