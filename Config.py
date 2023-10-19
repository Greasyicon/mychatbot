import os
os.environ['CURL_CA_BUNDLE'] = ''
token = os.environ.get('HUGGINGFACE_TOKEN')
if token:
    # Use the token for your operations
    print(f"-----______________--------- Hugging Face Token is set") #
else:
    print("WARNING ---- ______ -----Hugging Face Token not set or not found! "
          "May be Required to Download Model from Hugging Face hub.")

# colors for print
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

# To load a specific model, specify the model name:
import torch
# use the Quantized model is cuda is available
if torch.cuda.is_available():
    hf_model_repo = "TheBloke/Llama-2-7b-Chat-GPTQ" # "TheBloke/Llama-2-13B-GPTQ" #
else:
    hf_model_repo = "meta-llama/Llama-2-7b-chat-hf"

print(f"The model selected is : {hf_model_repo}")

db_cred = {
    'username' : 'llama',
    'password' : 'llama',
    'host' : 'localhost',
    'port' : '5432',
    'database' : 'llama',
    'schema' : 'public'
}
