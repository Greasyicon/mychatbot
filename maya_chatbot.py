# maya_chatbot.py

from Config import hf_model_repo, hf_model_repo_quant, token
import time
import myllm
from llama_index.llms.base import ChatMessage

def chatbot_response(user_input):
    messages = [ChatMessage(role="user", content=user_input)]
    return myllm.my_llm().chat(messages)
def run_local():
    # Local interaction with the chatbot
    while True:
        timeStart = time.time()
        user_input = input("You: ")
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

    mode = input("\n\nEnter 'web' to run on Flask or 'local' to run locally: ").strip().lower()

    if mode == "web":
        run_web()
    elif mode == "local":
        run_local()
    else:
        print("Invalid mode. Exiting.")
