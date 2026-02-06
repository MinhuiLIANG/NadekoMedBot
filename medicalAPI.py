import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import bitsandbytes as bnb
import transformers
from peft import PeftModel
from colorama import Fore, Style

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)

model_name = "Qwen/Qwen3-8B"
cache_dir = "./cache"
ckpt_dir = "./exp1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(ckpt_dir, exist_ok=True)

ckpts = []
for ckpt in os.listdir(ckpt_dir):
    if ckpt.startswith("checkpoint-"):
        ckpts.append(ckpt)

ckpts = sorted(ckpts, key=lambda ckpt: int(ckpt.split("-")[-1]))
ckpt_name = os.path.join(ckpt_dir, ckpts[-1])

max_len = 128
temperature = 0.1
top_p = 0.3
no_repeat_ngram_size = 3
seed = 42

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=nf4_config,
    device_map={'': 0},
    cache_dir=cache_dir
)
model = PeftModel.from_pretrained(model, ckpt_name, device_map={'': 0})
model.eval()
generation_config = GenerationConfig(
    do_sample=True,
    temperature=temperature,
    num_beams=1,
    top_p=top_p,
    # top_k=top_k,
    no_repeat_ngram_size=no_repeat_ngram_size,
)

instruction = """你是一位有经验的医生，根据患者的描述，提供专业的医疗建议。"""

@torch.no_grad()
def evaluate(input_text):
    verbose = True
    prompt = f"""<|im_start|>system
{instruction}
<|im_end|>
<|im_start|>user
{input_text}
<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    prompt_len = input_ids.shape[-1]

    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_len,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = generation_output.sequences[0][prompt_len:]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if verbose:
        print(output)

    return {"generation": output}