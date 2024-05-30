from transformers import BitsAndBytesConfig, pipeline
import torch

def load_LLaMa_model(model_name="meta-llama/Llama-2-7b-chat-hf", access_token=None, device_map="auto", quantization_config=True):
    assert 'Llama' in model_name, 'Please use meta-llama model'
    if quantization_config is None:
        llm_pipe = pipeline("text-generation",
            model=model_name,
            token=access_token,
            device_map=device_map)
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        llm_pipe = pipeline("text-generation",
            model=model_name,
            model_kwargs={
                "quantization_config": quantization_config
                },
            token=access_token,
            device_map=device_map)
    return llm_pipe
