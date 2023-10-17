import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import os
os.environ['CURL_CA_BUNDLE'] = ''
token = os.environ.get('HUGGINGFACE_TOKEN')
if token:
    # Use the token for your operations
    print(f"\n-----______________--------- Hugging Face Token is set\n") #
else:
    print("\nWARNING ---- ______ ----- Hugging Face Token not set or not found! May be Required to Download Model from Hugging Face hub.\n")
# from IPython.display import Markdown, display

import torch
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index import SQLDatabase

# To load a specific model, specify the model name:
hf_model_repo_quant = "TheBloke/Llama-2-7b-Chat-GPTQ" # "TheBloke/Llama-2-13B-GPTQ" #
hf_model_repo = "meta-llama/Llama-2-7b-chat-hf"

SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. 
Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- You are amazing at understanding the SQL database structure and always create the query (no matter how complicated 
the query is) which helps answering the user question.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
- Never repeat the question. Only generate the answer in a nice formatted manner.
"""

query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=500,
    # generate_kwargs={"temperature": 0.0, "do_sample": True},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=hf_model_repo_quant,
    model_name=hf_model_repo_quant,
    device_map="auto",
    # change these settings below depending on your GPU
    model_kwargs={"torch_dtype": torch.float16, "token":token}, #, "load_in_8bit": True
)

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en"
)
set_global_service_context(service_context)

# Get structured data
# Connect to Postgres Database
# Read blog: https://www.dataherald.com/blog/how-to-connect-llm-to-sql-database-with-llamaindex
from sqlalchemy import create_engine, MetaData
username = 'llama'
password = 'llama'
host = 'localhost'
port = '5432'
mydatabase = 'llama'
schema = 'public'
engine = create_engine(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}",
                       connect_args={'options': '-csearch_path={}'.format(schema)}, echo=False)

#load all table definitions
metadata_obj = MetaData()
metadata_obj.reflect(engine)
sql_database = SQLDatabase(engine)
table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = []
for table_name in metadata_obj.tables.keys():
    table_schema_objs.append(SQLTableSchema(table_name=table_name))

# Add Unstructured Data  -  context
# print(dir)
# load documents
documents = SimpleDirectoryReader(
    f"{os.getcwd()}\data"
).load_data()
# print(documents)
table_schema_objs.append(documents)
# We dump the table schema information into a vector index.
# The vector index is stored within the context builder for future use.
obj_index = ObjectIndex.from_objects(table_schema_objs, table_node_mapping, VectorStoreIndex,)
# index = VectorStoreIndex.from_documents(documents)#, service_context=service_context)

# query_engine = index.as_query_engine()
## Persist the index to disk
## index.storage_context.persist(persist_dir="index_storage")

# We construct a SQLTableRetrieverQueryEngine.
# Note that we pass in the ObjectRetriever so that we can dynamically retrieve the table during query-time.
# ObjectRetriever: A retriever that retrieves a set of query engine tools.
query_engine = SQLTableRetrieverQueryEngine(
    sql_database, obj_index.as_retriever(similarity_top_k=2), service_context=service_context,)

print("\n==============================================================================")
print("Entering into Q&A mode. Please enter - 'exit' anytime to close Q&A session.")
print("==============================================================================")
while(True):
    input_str = input('\nHow Can I help you?: ')
    # input_token_length = input('Enter length: ')

    if(input_str.lower() == 'exit'):
        break
    import time
    timeStart = time.time()
    try:
        response = query_engine.query(input_str)
        print("\nMayaAI: ", response)
        print(" \nMetadata Info:")
        print("     MayaSQL:", response.metadata['sql_query'])
        print("     MayaSQLResult:", response.metadata['result'])
    except Exception as e:
        print(f"ERROR --- SQL {e} Please modify the question so that question is only related to one table.")

    print("     Time taken: ", -timeStart + time.time())