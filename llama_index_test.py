import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import os
os.environ['CURL_CA_BUNDLE'] = ''
token = os.environ.get('HUGGINGFACE_TOKEN')
if token:
    # Use the token for your operations
    print(f"-----______________--------- Hugging Face Token is set") #
else:
    print("WARNING ---- ______ -----Hugging Face Token not set or not found! May be Required to Download Model from Hugging Face hub.")
# from IPython.display import Markdown, display

import torch
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM

# To load a specific model, specify the model name:
hf_model_repo_quant = "TheBloke/Llama-2-7b-Chat-GPTQ" # "TheBloke/Llama-2-13B-GPTQ" #
hf_model_repo = "meta-llama/Llama-2-7b-chat-hf"

SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. 
Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
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
    max_new_tokens=20,
    # generate_kwargs={"temperature": 0.0, "do_sample": True},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=hf_model_repo,
    model_name=hf_model_repo,
    device_map="auto",
    # change these settings below depending on your GPU
    model_kwargs={"torch_dtype": torch.float32, "token":token}, #, "load_in_8bit": True
)

dir = os.getcwd()
# print(dir)
# load documents
documents = SimpleDirectoryReader(
    f"{dir}\data"
).load_data()
print(documents)

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en"
)
set_global_service_context(service_context)

index = VectorStoreIndex.from_documents(documents)#, service_context=service_context)
# Persist the index to disk
# index.storage_context.persist(persist_dir="index_storage")

query_engine = index.as_query_engine()

while(True):
    input_str = input('Enter: ')
    # input_token_length = input('Enter length: ')

    if(input_str == 'exit'):
        break
    import time
    timeStart = time.time()

    response = query_engine.query(input_str)
    print(response)

    print("Time taken: ", -timeStart + time.time())