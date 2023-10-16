import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from IPython.display import Markdown, display

import torch
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM

# To load a specific model, specify the model name:
hf_model_repo_quant = "TheBloke/Llama-2-7b-Chat-GPTQ" # "TheBloke/Llama-2-13B-GPTQ" #


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
    max_new_tokens=2048,
    # generate_kwargs={"temperature": 0.0, "do_sample": True},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=hf_model_repo_quant,
    model_name=hf_model_repo_quant,
    device_map="auto",
    # change these settings below depending on your GPU
    model_kwargs={"torch_dtype": torch.float16}, #, "load_in_8bit": True
)

# load documents
documents = SimpleDirectoryReader(
    "C:\Projects\Llama2\data"
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