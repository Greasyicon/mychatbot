import torch
import time, os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Using llama-index Huggingface LLm pipeline
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from llama_index.llms.base import ChatMessage



########################################## CHAT BOT #######################################################
def chat_bot(user_input=None):
    print("\n==============================================================================")
    print("Entering into Q&A mode. Please enter - 'exit' anytime to close Q&A session.")
    print("==============================================================================")
    while (True):
        if user_input is None:
            user_input = input(f"{bcolors.OKCYAN}Enter your query here, Sire: {bcolors.ENDC}")
        # input_token_length = input('Enter output length expected (more length -> more response time): ')

        if (user_input == 'exit'):
            break

        timeStart = time.time()

        # prepare message for HF
        messages = [ChatMessage(role="user", content=user_input)]
        output_str_llama_index = llm.chat(messages)

        print(f"{bcolors.OKBLUE}Maya Chatbot {output_str_llama_index}{bcolors.ENDC}")

        print("Time taken: ", -timeStart + time.time())


########################################## DOCUMENT BOT #######################################################
def doc_bot():

    print("\n==============================================================================")
    print("Entering into Q&A mode. Please enter - 'exit' anytime to close Q&A session.")
    print("==============================================================================")
    while (True):
        user_input = input(f"\n{bcolors.OKCYAN}Enter your query here, Sire: {bcolors.ENDC}")
        # input_token_length = input('Enter length: ')

        if (user_input.lower() == 'exit'):
            break

        timeStart = time.time()

        response = query_engine.query(user_input)
        print(f"\n{bcolors.OKBLUE}Maya Chatbot assistant: {response}{bcolors.ENDC}")

        print("Time taken: ", -timeStart + time.time())

def run_local():
    ################################### Local Bot Mode ########################################################
    # chatbot mode
    bot_modes = ['chatbot', 'docbot']
    bot_mode_default = 'chatbot'  # 'docbot'

    bot_mode = input('Enter the bot mode (chtabot or docbot): ')
    if bot_mode not in bot_modes:
        bot_mode = bot_mode_default
        print(f"Bot Model not selected, so using default bot model {bot_mode_default}")

    ##### Local interaction with the chatbot
    ## Do General Q&A
    if (bot_mode == 'chatbot'):
        chat_bot()

    ## Read documents and do Q&A!
    if bot_mode == 'docbot':
        doc_bot()


def run_web():
    from flask import Flask, render_template, request, jsonify

    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')


    @app.route('/ask', methods=['POST'])
    def ask():
        user_input = request.json['message']
        mode = request.json['mode']

        if mode == 'chatbot':
            # Call your chatbot function here
            response = chat_bot(user_input)
        elif mode == 'docbot':
            # Call your docbot function here
            response = doc_bot(user_input)
        else:
            response = "Error: Invalid mode"

        return jsonify({'response': response})

    app.run(debug=True)


if __name__ == '__main__':
    # app.run(debug=True)

    # to run on cuda the following code is needed!
    os.environ['CURL_CA_BUNDLE'] = ''

    token = os.environ.get('HUGGINGFACE_TOKEN')
    if token:
        # Use the token for your operations
        print(f"-----______________--------- Hugging Face Token is set") #
    else:
        print("WARNING ---- ______ -----Hugging Face Token not set or not found! "
              "May be Required to Download Model from Hugging Face hub.")

    # use the Quantized model is cuda is available
    if torch.cuda.is_available():
        print("\nCuda is available! yay!")
        cuda_ind = True
        hf_model_repo = "TheBloke/Llama-2-7b-Chat-GPTQ" # "TheBloke/Llama-2-13B-GPTQ" #
        t_dtype = torch.float16 # data type to float16 for quantized models
    else:
        cuda_ind = False
        hf_model_repo = "meta-llama/Llama-2-7b-chat-hf"#, "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf"
        t_dtype = torch.float32 # data type to float32 or float16 for non-quantized models

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
    # Max output Tokens expected
    max_output_tokens = 100
    print(
        f"Max Output Tokens is {max_output_tokens}. Increasing max output tokens will increase the model response time. ")

    # Set the Hugging Face pipeline
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=max_output_tokens,
        # generate_kwargs={"temperature": 0.0, "do_sample": True},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=hf_model_repo,
        model_name=hf_model_repo,
        device_map="auto",
        # change these settings below depending on your GPU
        model_kwargs={"torch_dtype": t_dtype, "token": token},  # , "load_in_8bit": True
    )
    ## Seeting Query Engine - Llama Index for reading docs
    from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context

    print("Vectorizing and Indexing documents data. Patience!")
    # Get root directory
    dir = os.getcwd()
    # Read the documents from data folder in root dir
    # load documents
    documents = SimpleDirectoryReader(os.path.join(dir, "data")).load_data()
    # Set service context for indexing data
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model="local:BAAI/bge-small-en")
    set_global_service_context(service_context)
    # Create index
    index = VectorStoreIndex.from_documents(documents)  # , service_context=service_context)
    # Persist the index to disk
    # index.storage_context.persist(persist_dir="index_storage")
    # Create an engine to query the document
    query_engine = index.as_query_engine()

    ####### Web or Local Bot
    mode = input("\n\nEnter 'web' to run on Flask or 'local' to run locally: ").strip().lower()

    if mode == "web":
        run_web()
    elif mode == "local":
        run_local()
    else:
        print("Invalid mode. Exiting.")

    print("\nMaya Chatbot assistant: Bye Dear!")


######################## ALL DONE ############################################