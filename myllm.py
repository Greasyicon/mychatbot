import Config
import torch
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate


def my_llm():
    SYSTEM_PROMPT = """
    You are an AI assistant that answers questions in a friendly manner, based on the given source documents. 
            Here are some rules you always follow:
            - Generate human readable output, avoid creating output with gibberish text.
            - Generate only the requested output, don't include any other language before or after the requested output.
            - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
            - Generate professional language typically used in business documents in North America.
            - Never generate offensive or foul language.
            - Never repeat the question. Only generate the answer in a nice formatted manner.
            - Answer all questions you can.
            """
    query_wrapper_prompt = PromptTemplate(
        "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
    )
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=100,
        # generate_kwargs={"temperature": 0.0, "do_sample": True},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=Config.hf_model_repo,
        model_name=Config.hf_model_repo,
        device_map="auto",
        # change these settings below depending on your GPU
        model_kwargs={"torch_dtype": torch.float16, "token": Config.token},  # , "load_in_8bit": True
    )
    return llm

def maya_ai(query_engine):
    print("\n==============================================================================")
    print("Entering into Q&A mode. Please enter - 'exit' anytime to close Q&A session.")
    print("==============================================================================")
    while (True):
        input_str = input('\nHow Can I help you?: ')
        # input_token_length = input('Enter length: ')

        if (input_str.lower() == 'exit'):
            break
        import time
        timeStart = time.time()
        try:
            # composite_engine = CompositeQueryEngine(query_engine, index.as_query_engine())
            # response = composite_engine.composite_query(input_str)
            # response = composite_engine.unified_query(input_str)
            response = query_engine.query(input_str)
            print("\nMayaAI: ", response)
            try:
                print(" \nMetadata Info:")
                print("     MayaSQL:", response.metadata['sql_query'])
                print("     MayaSQLResult:", response.metadata['result'])
            except:
                pass
        except Exception as e:
            print(f"ERROR --- SQL {e} Please modify the question so that question is only related to one table.")

        print("     Time taken: ", -timeStart + time.time())