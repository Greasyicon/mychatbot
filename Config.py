import os
os.environ['CURL_CA_BUNDLE'] = ''
token = os.environ.get('HUGGINGFACE_TOKEN')
if token:
    # Use the token for your operations
    print(f"-----______________--------- Hugging Face Token is set") #
else:
    print("WARNING ---- ______ -----Hugging Face Token not set or not found! "
          "May be Required to Download Model from Hugging Face hub.")
# from IPython.display import Markdown, display


# To load a specific model, specify the model name:
hf_model_repo_quant = "TheBloke/Llama-2-7b-Chat-GPTQ" # "TheBloke/Llama-2-13B-GPTQ" #
hf_model_repo = "meta-llama/Llama-2-7b-chat-hf"

db_cred = {
    'username' : 'llama',
    'password' : 'llama',
    'host' : 'localhost',
    'port' : '5432',
    'database' : 'llama',
    'schema' : 'public'
}
