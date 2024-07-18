import torch, transformers, pyreft
import os

from .prompt import prompt_template

model_name = 'meta-llama/Llama-2-7b-chat-hf'
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map='cuda',
    cache_dir='./workspace', token=os.getenv('HF_TOKEN')
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name, model_max_tokens=2048, use_fast=False,
    padding_side="right", token=os.getenv('HF_TOKEN')
)
tokenizer.pad_token = tokenizer.unk_token

# Test case
prompt = prompt_template("render a data frame with oxen")
print(prompt)
tokens = tokenizer(prompt, return_tensors='pt').to('cuda')

# # Load the reft model
reft_model = pyreft.ReftModel.load('./trained_intervention', model)
reft_model.set_device('cuda')

# Generate a prediction
base_unit_position = tokens['input_ids'].shape[-1] -1
_, response = reft_model.generate(tokens,
                            unit_locations={'sources->base':(None, [[[base_unit_position]]])},
                            intervene_on_prompt=True
                            )
print(tokenizer.decode(response[0]))