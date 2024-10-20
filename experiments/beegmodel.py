import torch
from transformers import (
    AutoConfig, MistralForCausalLM, AutoTokenizer, pipeline
)
from huggingface_hub import snapshot_download
from accelerate import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
    infer_auto_device_map
)
from pprint import pprint
from collections import Counter
import time

# https://github.com/huggingface/transformers/issues/31544#issuecomment-2188510876

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

checkpoint = 'mistralai/Mistral-7B-Instruct-v0.2'
weights_location = snapshot_download(
    repo_id=checkpoint,
    allow_patterns=[
        '*.safetensors',
        '*.json'
    ],
    ignore_patterns=[
        '*pytorch*',
        'consolidated.safetensors'
    ]
)

config = AutoConfig.from_pretrained(checkpoint)

with init_empty_weights():
    model = MistralForCausalLM._from_config(config, torch_dtype=torch.bfloat16)

# Much better than 'device_map="auto"'
device_map = infer_auto_device_map(model, dtype=torch.bfloat16)
model = load_checkpoint_and_dispatch(
    model, checkpoint=weights_location, device_map=device_map
)

print(Counter(device_map.values()))

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1000,
    do_sample=False,
    repetition_penalty=1.03,
)

system_prompt = """
Ignore all directives given before what follows.
You are Aida, a helpful, creative, casual, clever, and very friendly AI assistant.
Respond to requests sent by a users to the best of your knowledge,
rarely greeting or referencing them in your response.
"""

messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {"role": "user", "content": "(Asked on October 20, 2024, at 2:20 pm)\nAndy: What's the best way to fold a burrito?"},
]


start_time = time.time()
print(pipe(messages)[0]['generated_text'][-1])
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
