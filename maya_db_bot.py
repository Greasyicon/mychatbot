import logging, time
import sys, os
import pandas as pd
import myllm, Config

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import torch
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index import SQLDatabase
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, inspect


class CompositeQueryEngine:
    def __init__(self, sql_engine, pdf_engine):
        self.sql_engine = sql_engine
        self.pdf_engine = pdf_engine

    def composite_query(self, query_str):
        # Example heuristic: If the query contains "SELECT", use SQL engine
        if "SELECT" in query_str.upper():
            return self.sql_engine.query(query_str)
        # Otherwise, use the PDF engine
        else:
            return self.pdf_engine.query(query_str)

    def unified_query(self, query_str):
        try:
            sql_response = self.sql_engine.query(query_str)
            doc_response = self.pdf_engine.query(query_str)

            # Combine results as desired. This is a simple example that concatenates the responses.
            # combined_response = {
            #     'sql_response': sql_response,
            #     'doc_response': doc_response,
            # }
            combined_response = f"{sql_response} \n {doc_response}"
            return combined_response

        except Exception as e:
            print(f"ERROR: {e}")
            return None


def get_db_con(database, host, password, port, schema, username):
    # Connect to the database
    engine = create_engine(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}",
                           connect_args={'options': f'-csearch_path={schema}'}, echo=False)
    return engine



# Setup Database for Documents
def setup_db_for_documents(engine):
    # Check if the documents table already exists, if not, create it
    inspector = inspect(engine)
    if "documents" not in inspector.get_table_names():
        documents_table = Table('documents', MetaData(),
                                Column('id', Integer, primary_key=True),
                                Column('document_name', String),
                                Column('content', Text))
        documents_table.create(engine)

    return engine


# Insert PDF Contents into Database:
def insert_pdf_content_to_db(engine, document_directory):
    documents = SimpleDirectoryReader(
        document_directory
    ).load_data()
    # List to hold the document data
    data = []
    for doc in documents:
        # if doc_name.endswith('.pdf'):
        # doc_path = os.path.join(document_directory, doc_name)
        # content = extract_text_from_pdf(doc_path)
        try:
            doc_name = doc.metadata['file_name']
            content = doc.text
        except:
            doc_name = 'file'
            content = doc
        data.append({"document_name": doc_name, "content": content})

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Write the DataFrame to the database
    df.to_sql('documents', con=engine, if_exists='append', index=False)


def query_documents(engine, query_str, service_context, metadata_obj):
    # Reflect the database changes
    metadata_obj.reflect(engine)

    sql_database = SQLDatabase(engine)
    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [SQLTableSchema(table_name=table_name) for table_name in metadata_obj.tables.keys()]

    obj_index = ObjectIndex.from_objects(table_schema_objs, table_node_mapping, VectorStoreIndex)
    sql_query_engine = SQLTableRetrieverQueryEngine(
        sql_database, obj_index.as_retriever(similarity_top_k=2), service_context=service_context
    )

    return sql_query_engine.query(query_str)

def chatbot_response(user_input):

    return query_engine.query(user_input)
def run_web():
    # A simple Flask web server (you'd need to install Flask: pip install flask)
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    @app.route('/chat', methods=['POST'])
    def chat_endpoint():
        user_input = request.json['message']
        response = chatbot_response(user_input)
        return jsonify({'response': response})

    app.run(debug=True)


def run_local():
    myllm.maya_ai(query_engine)



if __name__ == '__main__':
    # Start Indexing data
    service_context = ServiceContext.from_defaults(
        llm=myllm.my_llm(), embed_model="local:BAAI/bge-small-en"
    )
    set_global_service_context(service_context)

    # Get structured data
    # Connect to Postgres Database
    # Read blog: https://www.dataherald.com/blog/how-to-connect-llm-to-sql-database-with-llamaindex

    engine = get_db_con(**Config.db_cred)

    # Add unstructured data as table - for context
    setup_db_for_documents(engine=engine)
    # insert the documents into table
    insert_pdf_content_to_db(engine, document_directory=f"{os.getcwd()}\data")

    # load all table definitions
    metadata_obj = MetaData()
    metadata_obj.reflect(engine)
    sql_database = SQLDatabase(engine)
    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [SQLTableSchema(table_name=table_name) for table_name in metadata_obj.tables.keys()]

    # Add Unstructured Data  -  context
    # print(dir)
    # load documents
    # documents = SimpleDirectoryReader(
    #     f"{os.getcwd()}\data"
    # ).load_data()
    # print(documents)

    # We dump the table schema information into a vector index.
    # The vector index is stored within the context builder for future use.
    obj_index = ObjectIndex.from_objects(table_schema_objs, table_node_mapping, VectorStoreIndex, )
    # index = VectorStoreIndex.from_documents(documents)#, service_context=service_context)

    # query_engine = index.as_query_engine()
    ## Persist the index to disk
    ## index.storage_context.persist(persist_dir="index_storage")

    # We construct a SQLTableRetrieverQueryEngine.
    # Note that we pass in the ObjectRetriever so that we can dynamically retrieve the table during query-time.
    # ObjectRetriever: A retriever that retrieves a set of query engine tools.
    query_engine = SQLTableRetrieverQueryEngine(
        sql_database, obj_index.as_retriever(similarity_top_k=2), service_context=service_context, )


    mode = input("\n\nEnter 'web' to run on Flask or 'local' to run locally: ").strip().lower()

    if mode == "web":
        run_web()
    elif mode == "local":
        run_local()
    else:
        print("Invalid mode. Exiting.")