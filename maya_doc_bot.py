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

def read_confluence():
    from llama_hub.confluence import ConfluenceReader

    token = {
        "access_token": "ATOAdX1C386QZRWLw46E_NRy3y8MjfdU6j2pLietZpqi_m3UuHU3BtcXIBmyYu_-lshj1A57A3CB",
        "token_type": "bearer"
    }
    oauth2_dict = {
        "client_id": "1ZOgqvjvFCjdsX2qdZlZ8Bs2RzYvBmwt",
        "token": token
    }

    base_url = "https://bouncybear.atlassian.net/wiki"

    # page_ids = ["<page_id_1>", "<page_id_2>", "<page_id_3"]
    space_key = "~712020d26a0bf843a04a54b0e4c6254eb599ec"

    reader = ConfluenceReader(base_url=base_url)#, oauth2=oauth2_dict)
    documents = reader.load_data(space_key=space_key, include_attachments=True, page_status="current")
    print(documents)
    # documents.extend(reader.load_data(page_ids=page_ids, include_children=True, include_attachments=True))
    return documents

if __name__ == '__main__':
    read_confluence()
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