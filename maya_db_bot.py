# import logging, time
import sys, os
os.environ['CURL_CA_BUNDLE'] = ''
import pandas as pd
import Config
import myllm

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import torch
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context, SQLDatabase
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index.indices.vector_store import VectorIndexAutoRetriever
from llama_index.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from sqlalchemy import create_engine, MetaData#, Table, Column, Integer, String, Text, inspect
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.tools import ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine, SQLJoinQueryEngine, RetrieverQueryEngine
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.selectors.llm_selectors import (
    LLMSingleSelector,
    LLMMultiSelector,
)
from llama_index.query_engine.router_query_engine import RouterQueryEngine
# import nest_asyncio
# nest_asyncio.apply()

def get_db_con(database, host, password, port, schema, username):
    # Connect to the database
    engine = create_engine(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}",
                           connect_args={'options': f'-csearch_path={schema}'}, echo=False)
    return engine

def chatbot_response(user_input):
    return query_engine.query(user_input)

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


def run_local():
    myllm.maya_ai(query_engine)



if __name__ == '__main__':

    # Using the LlamaDebugHandler to print the trace of the sub questions
    # captured by the SUB_QUESTION callback event type
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    service_context = ServiceContext.from_defaults(llm=myllm.my_llm(), embed_model="local:BAAI/bge-small-en",
                                                callback_manager=callback_manager)
    # set_global_service_context(service_context)

    # Get structured data
    # Connect to Postgres Database
    # Read blog: https://www.dataherald.com/blog/how-to-connect-llm-to-sql-database-with-llamaindex

    engine = get_db_con(**Config.db_cred)

    # # Add unstructured data as table - for context
    # setup_db_for_documents(engine=engine)
    # # insert the documents into table
    # insert_pdf_content_to_db(engine, document_directory=f"{os.getcwd()}\data")

    # Start Indexing data
    # first load all SQL DB table definitions
    metadata_obj = MetaData()
    metadata_obj.reflect(engine)
    sql_database = SQLDatabase(engine)
    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [SQLTableSchema(table_name=table_name) for table_name in metadata_obj.tables.keys()]

    # We dump the table schema information into a vector index.
    # The vector index is stored within the context builder for future use.
    obj_index = ObjectIndex.from_objects(table_schema_objs, table_node_mapping, VectorStoreIndex, )

    # query_engine = index.as_query_engine()
    ## Persist the index to disk
    ## index.storage_context.persist(persist_dir="index_storage")
    #https://gpt-index.readthedocs.io/en/latest/examples/query_engine/SQLJoinQueryEngine.html
    # We construct a SQLTableRetrieverQueryEngine.
    # Note that we pass in the ObjectRetriever so that we can dynamically retrieve the table during query-time.
    # ObjectRetriever: A retriever that retrieves a set of query engine tools.
    db_query_engine = SQLTableRetrieverQueryEngine(
        sql_database, obj_index.as_retriever(similarity_top_k=1), service_context=service_context, )

    # NLSQLTableQueryEngine uses llama CPP and then uses 13b GGUF model which is not what we want
    # db_query_engine = NLSQLTableQueryEngine(
    #     sql_database=sql_database#, llm=myllm.my_llm()
    #     # tables=["city_stats"],
    # )


    # Add Unstructured Data  -  context
    # print(dir)
    # load documents
    documents = SimpleDirectoryReader(f"{os.getcwd()}\data").load_data()
    doc_index = VectorStoreIndex.from_documents(documents, use_async=True)#, service_context=service_context)
    doc_query_engine = doc_index.as_query_engine()

    doc_query_engine_tool = QueryEngineTool(
            query_engine=doc_query_engine,
            metadata=ToolMetadata(
                name="Paul Graham Essay",
                description="Paul Graham essay on What I Worked On",
            ),
        )

    db_query_engine_tool = QueryEngineTool(
            query_engine=db_query_engine,
            metadata=ToolMetadata(
                name="Postgres Data",
                description="Data on Different customer and orders",
            ),
        )

    ## Setup Query Engine Tools
    query_engine_tools = [db_query_engine_tool, doc_query_engine_tool]


    # todo: the SQLJoinQueryEngine and SubQuestionQueryEngine is not working with Llama 2 but works
    #  with OpenAI which is default LLM

    #  SQLJoinQueryEngine lets us pick the best query engine when there are multiple query engines
    query_engine = SQLJoinQueryEngine(
        db_query_engine_tool, doc_query_engine_tool,  service_context=service_context
    )
    # response = query_engine.query("Who purchased Laptop?")
    # SubQuestionQueryEngine lets us break the complex query into multiple queries and then combine the response
    # query_engine = SubQuestionQueryEngine.from_defaults(
    #     query_engine_tools=query_engine_tools,
    #     service_context=service_context,
    #     use_async=True,
    # )
    # RouterQueryEngine lets us select one out of several candidate query engines to execute a query.
    # This is also using GGUF model and not the one we select
    # query_engine = RouterQueryEngine( #db_query_engine
    #     selector=LLMMultiSelector.from_defaults(),
    #     query_engine_tools=query_engine_tools,
    #     # summarizer=tree_summarize,
    # )

    mode = input("\n\nEnter 'web' to run on Flask or 'local' to run locally: ").strip().lower()

    if mode == "web":
        run_web()
    elif mode == "local":
        run_local()
    else:
        print("Invalid mode. Exiting.")