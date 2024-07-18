import torch, transformers, pyreft
import pandas as pd
import os
from transformers import TextStreamer

from prompt import prompt_template

model_name = 'meta-llama/Llama-2-7b-chat-hf'
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map='cuda',
    token=os.getenv('HF_TOKEN')
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name, model_max_tokens=2048, use_fast=False,
    padding_side="right", token=os.getenv('HF_TOKEN')
)
tokenizer.pad_token = tokenizer.unk_token
streamer = TextStreamer(tokenizer)

# Test case
# prompt = prompt_template("how to render a data frame with oxen")
# print(prompt)
# tokens = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
# response = model.generate(tokens, max_new_tokens=256, streamer=streamer)
# print(tokenizer.decode(response[0]))

# Get the reft model
reft_config = pyreft.ReftConfig(
    representations={
        "layer":15,
        "component":"block_output",
        # "component": "model.layers[0].output",
        "low_rank_dimension":4,
        "intervention":pyreft.LoreftIntervention(
            embed_dim=model.config.hidden_size, low_rank_dimension=4
        )
    }
)

reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device('cuda')

# GRAB Data
df = pd.read_json('train.jsonl', lines=True)
X = df['prompt'].values
y = df['response'].values

# Operate on last token
data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer,
    model,
    [prompt_template(x) for x in X],
    y
)

# Training arguments
training_arguments = transformers.TrainingArguments(
    num_train_epochs=50,
    output_dir='./models',
    per_device_train_batch_size=2,
    learning_rate=4e-3,
    logging_steps=20,
    report_to=[]
)

# Trainer for the reft model
trainer = pyreft.ReftTrainerForCausalLM(
    model=reft_model,
    tokenizer=tokenizer,
    args=training_arguments,
    **data_module
)

# Train the model!!
_ = trainer.train()

# Save the model
reft_model.set_device('cpu')
reft_model.save(
    save_directory='./reft_intervention_model'
)