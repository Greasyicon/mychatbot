# maya_chatbot.py

import Config
import time
import myllm
from llama_index.llms.base import ChatMessage
from flask import Flask, render_template, request
from flask_socketio import SocketIO

# Create the Flask app and SocketIO instance
app = Flask(__name__)
socketio = SocketIO(app)

def chatbot_response(user_input):
    print ("User input is -", user_input)
    messages = [ChatMessage(role="user", content=user_input)]
    return llm.chat(messages)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
@socketio.on('message')
def handle_message(data):
    message = data['message']
    mode = data['mode']
    timeStart = time.time()
    response = chatbot_response(message)

    print(f"{Config.bcolors.OKBLUE}Maya Chatbot {response}{Config.bcolors.ENDC}")

    print("Time taken: ", -timeStart + time.time())
    socketio.emit('message', {'user': 'You', 'text': message})
    socketio.emit('message', {'user': 'Maya', 'text': response})


def run_local():
    # Local interaction with the chatbot
    while (True):
        user_input = input(f"{Config.bcolors.OKCYAN}Enter your query here, Sire: {Config.bcolors.ENDC}")
        # input_token_length = input('Enter output length expected (more length -> more response time): ')

        if (user_input == 'exit'):
            break

        timeStart = time.time()

        response = chatbot_response(user_input)

        print(f"{Config.bcolors.OKBLUE}Maya Chatbot {response}{Config.bcolors.ENDC}")

        print("Time taken: ", -timeStart + time.time())

def run_web():
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)
# def run_web():
#     # A simple Flask web server (you'd need to install Flask: pip install flask)
#     from flask import Flask, request, jsonify, render_template
#     app = Flask(__name__)
#     @app.route('/')
#     def index():
#         return render_template('index.html')
#     @app.route('/chat', methods=['POST'])
#     def chat_endpoint():
#         user_input = request.json['message']
#         response = chatbot_response(user_input)
#         return jsonify({'response': response})
#
#     app.run(debug=False)



if __name__ == "__main__":

    llm = myllm.my_llm()

    mode = input("\n\nEnter 'web' to run on Flask or 'local' to run locally: ").strip().lower()

    if mode == "web":
        run_web()
    elif mode == "local":
        run_local()
    else:
        print("Invalid mode. Exiting.")
