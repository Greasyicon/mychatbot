from llama_cpp import Llama

LLM = Llama(model_path="C:\Projects\Llama2\llama model\llama-2-13b-chat.Q6_K.gguf", f16_kv=True)

output = LLM("Hello, sup", max_tokens=256)