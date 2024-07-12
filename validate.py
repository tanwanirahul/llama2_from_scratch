'''
    Validates the implementation of our Llama2 model with the HF model.
'''
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from huggingface_hub import login
from config import LlamaConfig
from models import LlamaWithLMHead
import torch
import os
from safetensors import torch as safetorch
from transformers.utils.hub import cached_file
import json

def do_login():
    '''
        Loging to HF to access gated repo - Llama2
    '''
    access_token = os.environ.get("HF_ACCESS_TOKEN")
    login(token=access_token)


def load_output_as_tensor(model_name):
    '''
        loads the output of the given `model_name`
        and returns the logits as torch.tensor.
    '''
    input_filename = f"{model_name}_output.json"
    
    with open(input_filename, 'r') as f:
        output = json.loads(f.read())

        return torch.tensor(output["output"], dtype=torch.float32)

def run_eval_and_save_output(model, inputs, model_name):
    '''
        Runs the evaluation on the `model` for the given `inputs`
        and saves the output of the last token logits into
        the `model_name`_output in the json format.
    '''
    output = model(**inputs)
    output = output.logits

    print(f"output shape from model {model_name}: {output.shape}")
    print(f"persisting output logits of the last token:")    

    output_filename = f"{model_name}_output.json"
    last_token_logits = output[0][-1].tolist()
    
    with open(output_filename, 'w') as f:
        f.write(json.dumps({"output": last_token_logits}))
    
    print(f"saved output of {model_name} to {output_filename}.")

if __name__ == "__main__":


    hf_model_name = "meta-llama/Llama-2-7b-hf"
    
    # Prepare inputs to run the evaluations on.
    print(f"preparing inputs to run evaluations on.")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    text_inputs = "The purpose of life is to, "
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(text_inputs, return_tensors="pt")
    
    # Run the inputs through HF_model.
    hf_impln = "hf_llama2_impln"
    print(f"loading huggingface's llama2 implementation.")
    torch.manual_seed(101)
    torch.cuda.manual_seed(101)
    do_login()
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16)
    run_eval_and_save_output(model, inputs, hf_impln)
    model = None

    # run the inputs through our implementation of Llama2 and compare the outputs.
    our_impln = "our_llama2_impln"
    print(f"loading our llama2 implementation with pre-trained weights from huggingface.")
    torch.manual_seed(101)
    torch.cuda.manual_seed(101)
    safe_tensor_checkpoints = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]    
    model = LlamaWithLMHead.from_checkpoint(hf_model_name, safe_tensor_checkpoints)
    run_eval_and_save_output(model, inputs, our_impln)
    model = None
    
    print(f"comparing outputs from both the models.")
    # load the output and compare the results.
    hf_output_tensor = load_output_as_tensor(hf_impln)
    our_output_tensor = load_output_as_tensor(our_impln)

    do_outputs_match = torch.equal(hf_output_tensor, our_output_tensor)
    
    print(f"\noutputs matching?: {do_outputs_match}")