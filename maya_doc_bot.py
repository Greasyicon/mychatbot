import logging, time
import sys, os
import myllm, Config
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context

from flask import Flask, render_template, request, jsonify

def chatbot_response(user_input):

    return query_engine.query(user_input)
def run_web():
    # A simple Flask web server (you'd need to install Flask: pip install flask)

    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/ask', methods=['POST'])
    def ask():
        user_input = request.form['user_input']
        try:
            response = chatbot_response(user_input)
            return jsonify({"response": response})  # , "metadata": metadata})
        except Exception as e:
            return jsonify({"error": f"ERROR --- {e}."})

    app.run(debug=True)


def run_local():
    myllm.maya_ai(query_engine)

if __name__ == '__main__':

    dir = os.getcwd()

    # load documents
    documents = SimpleDirectoryReader(os.path.join(dir, "data")).load_data()

    service_context = ServiceContext.from_defaults(
        llm=myllm.my_llm(), embed_model="local:BAAI/bge-small-en")

    set_global_service_context(service_context)

    index = VectorStoreIndex.from_documents(documents)  # , service_context=service_context)
    # Persist the index to disk
    # index.storage_context.persist(persist_dir="index_storage")

    query_engine = index.as_query_engine()

    mode = input("\n\nEnter 'web' to run on Flask or 'local' to run locally: ").strip().lower()

    if mode == "web":
        run_web()
    elif mode == "local":
        run_local()
    else:
        print("Invalid mode. Exiting.")