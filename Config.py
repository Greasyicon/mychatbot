import os
os.environ['CURL_CA_BUNDLE'] = ''

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

token = os.environ.get('HUGGINGFACE_TOKEN')
if token:
    # Use the token for your operations
    print(f"-----______--------- Hugging Face Token is set") #
else:
    print(f"{bcolors.WARNING}WARNING ---- ______ -----Hugging Face Token not set or not found! "
          f"May be Required to Download Model from Hugging Face hub.{bcolors.ENDC}")
# To load a specific model, specify the model name:
import torch

# use the Quantized model is cuda is available
if torch.cuda.is_available():
    print("\nCuda is available! yay! Speeding...!")
    cuda_ind = True
    hf_model_repo = "TheBloke/Llama-2-7b-Chat-GPTQ" # "TheBloke/Llama-2-13B-GPTQ" #
    t_dtype = torch.float16 # data type to float16 for quantized models
    max_new_tokens = 500
else:
    cuda_ind = False
    hf_model_repo = "meta-llama/Llama-2-7b-chat-hf"#, "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf"
    t_dtype = torch.float32 # data type to float32 or float16 for non-quantized models
    max_new_tokens = 20 # slow response without cuda


print(f"The model selected is : {hf_model_repo}")

db_cred = {
    'username' : 'llama',
    'password' : 'llama',
    'host' : 'localhost',
    'port' : '5432',
    'database' : 'llama',
    'schema' : 'public'
}
