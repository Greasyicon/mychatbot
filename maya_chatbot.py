# maya_chatbot.py

import Config
import time
import myllm
from llama_index.llms.base import ChatMessage

def chatbot_response(user_input):
    messages = [ChatMessage(role="user", content=user_input)]
    return llm.chat(messages)

def run_local():
    # Local interaction with the chatbot
    while True:
        timeStart = time.time()
        user_input = input("Enter: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = chatbot_response(user_input)
        print(f"Maya Chatbot {response}")
        print("Time taken: ", -timeStart + time.time())

def run_web():
    # A simple Flask web server (you'd need to install Flask: pip install flask)
    from flask import Flask, request, jsonify, render_template
    app = Flask(__name__)
    @app.route('/')
    def index():
        return render_template('index.html')
    @app.route('/chat', methods=['POST'])
    def chat_endpoint():
        user_input = request.json['message']
        response = chatbot_response(user_input)
        return jsonify({'response': response})

    app.run(debug=True)



if __name__ == "__main__":

    llm = myllm.my_llm()

    mode = input("\n\nEnter 'web' to run on Flask or 'local' to run locally: ").strip().lower()

    if mode == "web":
        run_web()
    elif mode == "local":
        run_local()
    else:
        print("Invalid mode. Exiting.")
