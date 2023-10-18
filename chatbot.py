# from langchain.chains import LLMChain, SequentialChain
# from langchain.memory import ConversationBufferMemory
# from langchain import HuggingFacePipeline
# from langchain import PromptTemplate,  LLMChain
#
#
# from transformers import AutoModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM #, GPTQConfig

import json
import textwrap, time

timeStart = time.time()

import os
os.environ['CURL_CA_BUNDLE'] = ''
token = os.environ.get('HUGGINGFACE_TOKEN')
if token:
    # Use the token for your operations
    print(f"Token is: {token}")
else:
    print("Token not set or not found!")

hf_model_repo_quant = "TheBloke/Llama-2-7b-Chat-GPTQ" # "TheBloke/Llama-2-13B-GPTQ" #
hf_model_repo = "meta-llama/Llama-2-7b-chat-hf"#, "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf"


tokenizer = AutoTokenizer.from_pretrained(hf_model_repo, use_fast=True, token=token)

quantizaton_ind = True

if quantizaton_ind:
    # # Set quantization configuration
    # quantization_config = GPTQConfig(
    #  bits=4,
    #  group_size=128,
    #  dataset="c4",
    #  desc_act=False,
    #  tokenizer=tokenizer
    # )
    # # Load the model from HF
    # model = AutoModelForCausalLM.from_pretrained(hf_model_repo,
    #  quantization_config=quantization_config, device_map='auto')

    # Load the model from HF
    model = AutoModelForCausalLM.from_pretrained(hf_model_repo_quant, token=token)
                                                # torch_dtype = torch.float16,
                                                # device_map = "auto",
                                                # load_in_4bit=True,
                                                 # bnb_4bit_quant_type="nf4",
                                                 # bnb_4bit_compute_dtype=torch.float16

else:
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_repo,
        device_map='auto',
        torch_dtype=torch.float32,
        token=token
    )
# model.to('cuda')
print("Load model time: ", -timeStart + time.time())

# from transformers import pipeline
#
# pipe = pipeline("text-generation",
#                 model=model,
#                 tokenizer=tokenizer,
#                 torch_dtype=torch.float16,
#                 device_map="auto",
#                 max_new_tokens = 512,
#                 do_sample=True,
#                 top_k=30,
#                 num_return_sequences=1,
#                 eos_token_id=tokenizer.eos_token_id
#                 )
# B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<>\n", "\n<>\n\n"
# DEFAULT_SYSTEM_PROMPT = """\
# You are an advanced Life guru and mental health expert that excels at giving advice.
# Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical,
# racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased
# and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of
# answering something not correct. If you don't know the answer to a question, please don't share false information.
# Just say you don't know and you are sorry!"""
#
# def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT, citation=None):
#     SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
#     prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
#
#     if citation:
#         prompt_template += f"\n\nCitation: {citation}"  # Insert citation here
#
#     return prompt_template
#
# def cut_off_text(text, prompt):
#     cutoff_phrase = prompt
#     index = text.find(cutoff_phrase)
#     if index != -1:
#         return text[:index]
#     else:
#         return text
#
# def remove_substring(string, substring):
#     return string.replace(substring, "")
#
# def generate(text, citation=None):
#     prompt = get_prompt(text, citation=citation)
#     inputs = tokenizer(prompt, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model.generate(**inputs,
#                                  max_length=512,
#                                  eos_token_id=tokenizer.eos_token_id,
#                                  pad_token_id=tokenizer.eos_token_id,
#                                  )
#         final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#         final_outputs = cut_off_text(final_outputs, '')
#         final_outputs = remove_substring(final_outputs, prompt)
#
#     return final_outputs
#
# def parse_text(text):
#     wrapped_text = textwrap.fill(text, width=100)
#     print(wrapped_text + '\n\n')
#
# llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0.7,'max_length': 256, 'top_k' :50})
#
# system_prompt = "You are an advanced Life guru and mental health expert that excels at giving advice. "
# instruction = "Convert the following input text from a stupid human to a well-reasoned and step-by-step throughout advice:\n\n {text}"
# template = get_prompt(instruction, system_prompt)
# print(template)
#
# prompt = PromptTemplate(template=template, input_variables=["text"])
#
# llm_chain = LLMChain(prompt=prompt, llm=llm, verbose = False)
#
# text = "My life sucks, what do you suggest? Please don't tell me to medidate"
#
# response = llm_chain.run(text)
# print(response)
#
while(True):
    input_str = input('Enter: ')
    input_token_length = input('Enter output length expected (more length -> more latency): ')

    if(input_str == 'exit'):
        break

    timeStart = time.time()

    inputs = tokenizer.encode(
        input_str,
        return_tensors="pt"
    ).to('cuda')

    outputs = model.generate(
        inputs,
        max_new_tokens=int(input_token_length),
    )

    output_str = tokenizer.decode(outputs[0])

    print(output_str)

    print("Time taken: ", -timeStart + time.time())