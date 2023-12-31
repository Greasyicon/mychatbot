import Config
import torch
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate

# from llama_index.query_engine import RetryQueryEngine
# from llama_index.evaluation import RelevancyEvaluator
#
# query_response_evaluator = RelevancyEvaluator()
from llama_index.llms import OpenAI
def my_llm():
    # Set the system prompt
    SYSTEM_PROMPT = """
        You are Maya AI an AI assistant that answers questions in a friendly manner, based on the given source documents. 
        Here are some rules you always follow:
        - Your name is Maya AI. So always use Maya AI whenever you need to use your name.
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
        max_new_tokens=Config.max_new_tokens,
        # generate_kwargs={"temperature": 0.0, "do_sample": True},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=Config.hf_model_repo,
        model_name=Config.hf_model_repo,
        device_map="auto",
        # change these settings below depending on your GPU
        model_kwargs={"torch_dtype": Config.t_dtype, "token": Config.token},  # , "load_in_8bit": True
    )
    return llm#OpenAI(temperature=0, model="text-davinci-003", max_tokens=20)

def maya_ai(query_engine):
    print("\n==============================================================================")
    print("Entering into Q&A mode. Please enter - 'exit' anytime to close Q&A session.")
    print("==============================================================================")
    while (True):
        user_input = input(f"\n{Config.bcolors.OKCYAN}Enter your query here, Sire: {Config.bcolors.ENDC}")
        # input_token_length = input('Enter length: ')

        if (user_input.lower() == 'exit'):
            break
        import time
        timeStart = time.time()
        try:
            # composite_engine = CompositeQueryEngine(query_engine, index.as_query_engine())
            # response = composite_engine.composite_query(input_str)
            # response = composite_engine.unified_query(input_str)
            response = query_engine.query(user_input)

            # retry_query_engine = RetryQueryEngine(
            #     query_engine, query_response_evaluator
            # )
            # retry_response = retry_query_engine.query(user_input)
            # print(retry_response)

            print(f"\n{Config.bcolors.OKBLUE}Maya Chatbot assistant: {response}{Config.bcolors.ENDC}")
            try:
                print(" \nMetadata Info:")
                print("     MayaSQL:", response.metadata['sql_query'])
                print("     MayaSQLResult:", response.metadata['result'])
            except:
                pass
        except Exception as e:
            print(f"ERROR --- SQL {e} Please modify the question so that question is only related to one table.")

        print("     Time taken: ", -timeStart + time.time())