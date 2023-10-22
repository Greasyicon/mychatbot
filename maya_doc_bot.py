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

    base_url = "https://bouncybear.atlassian.net"

    page_ids = ['393222']
    space_key = "~712020d26a0bf843a04a54b0e4c6254eb599ec"

    reader = ConfluenceReader(base_url=base_url)#, oauth2=oauth2_dict)
    # documents = reader.load_data(page_ids=page_ids, include_children=False, include_attachments=True)
    documents = reader.load_data(space_key=space_key, include_attachments=False, page_status="current")
    # documents.extend(reader.load_data(page_ids=page_ids, include_children=True, include_attachments=True))

    ## Using confluence reader to read
    # from atlassian import Confluence
    #
    # # Initialize the Confluence instance
    # confluence = Confluence(
    #     url=base_url,
    #     username=os.getenv('CONFLUENCE_USERNAME'),
    #     password=os.getenv('CONFLUENCE_API_TOKEN'),
    #     cloud=True  # Set to True if you are using Confluence Cloud
    # )
    #
    # # Get page content
    # for page_id in page_ids:
    #     content = confluence.get_page_by_id(page_id, expand='body.storage')
    #     # Print page title and content
    #     print("Title:", content['title'])
    #     print("Content:", content['body']['storage']['value'])

    return documents

if __name__ == '__main__':


    # load documents
    #  From local Directory
    dir = os.getcwd()
    local_documents = SimpleDirectoryReader(os.path.join(dir, "data")).load_data()
    # From Confluence
    confluence_documents = read_confluence()
    # Combine documents
    documents = local_documents + confluence_documents
    service_context = ServiceContext.from_defaults(
        llm=myllm.my_llm(), embed_model="local:BAAI/bge-small-en")

    set_global_service_context(service_context)
    # indexing documents
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