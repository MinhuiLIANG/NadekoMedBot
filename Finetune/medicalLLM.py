import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
import argparse
import json
import warnings
import logging
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, load_from_disk
import transformers
from peft import PeftModel
from colorama import Fore, Style

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training
)

def generate_training_data(data_point):
    prompt = f"""<|im_start|>system
{data_point["instruction"]}
<|im_end|>
<|im_start|>user
{data_point["input"]}
<|im_end|>
<|im_start|>assistant
"""

    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        add_special_tokens=False,
    )["input_ids"]

    output_ids = tokenizer(
        data_point["output"] + tokenizer.eos_token,
        truncation=True,
        max_length=CUTOFF_LEN - len(prompt_ids),
        add_special_tokens=False,
    )["input_ids"]

    input_ids = prompt_ids + output_ids

    labels = [-100] * len(prompt_ids) + output_ids

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def evaluate(instruction, generation_config, max_len, input_text="", verbose=True):
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
    )

    generated_ids = generation_output.sequences[0][prompt_len:]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if verbose:
        print(output)

    return output

model_name = "Qwen/Qwen3-8B"

cache_dir = "./cache"

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=nf4_config,
    low_cpu_mem_usage=True
)

logging.getLogger('transformers').setLevel(logging.ERROR)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir
)
max_len = 128
generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.1,
    num_beams=1,
    top_p=0.3,
    no_repeat_ngram_size=3,
)

num_train_data = 1040

output_dir = "./output"
ckpt_dir = "./exp1"
num_epoch = 3
LEARNING_RATE = 3e-4

cache_dir = "./cache"
from_ckpt = False
ckpt_name = None
dataset_dir = "./tmp_medical.json"
logging_steps = 20
save_steps = 65
save_total_limit = 3
report_to = "none"
MICRO_BATCH_SIZE = 4
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
CUTOFF_LEN = 256
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 0
TARGET_MODULES = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

if from_ckpt:
    model = PeftModel.from_pretrained(model, ckpt_name)

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

data = load_dataset('json', data_files=dataset_dir, download_mode="force_redownload")
train_data = (
    data["train"]
    .select(range(3000))
    .shuffle(seed=42)
    .map(generate_training_data)
)

with open(dataset_dir, "r", encoding="utf-8") as f:
    test_datas = json.load(f)
test_datas = test_datas[3000:3050]
val_data = None

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=50,
        num_train_epochs=num_epoch,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=ckpt_dir,
        save_total_limit=save_total_limit,
        ddp_find_unused_parameters=False if ddp else None,
        report_to=report_to,
    ),
    data_collator=data_collator,
)

model.config.use_cache = False

if torch.__version__ >= "2" and sys.platform != 'win32':
    model = torch.compile(model)

trainer.train()

model.save_pretrained(ckpt_dir)

print("\n 如果上方有关于缺少键的警告，请忽略 :)")

ckpts = []
for ckpt in os.listdir(ckpt_dir):
    if ckpt.startswith("checkpoint-"):
        ckpts.append(ckpt)

ckpts = sorted(ckpts, key=lambda ckpt: int(ckpt.split("-")[-1]))
print("所有可用的 checkpoints：")
print(" id: checkpoint 名称")
for (i, ckpt) in enumerate(ckpts):
    print(f"{i:>3}: {ckpt}")

id_of_ckpt_to_use = -1

ckpt_name = os.path.join(ckpt_dir, ckpts[id_of_ckpt_to_use])
max_len = 128
temperature = 0.1
top_p = 0.3

import gc

if 'trainer' in locals():
  del trainer
if 'train_data' in locals():
  del train_data
if 'val_data' in locals() and val_data is not None:
  del val_data
if 'data' in locals():
  del data

del model
del tokenizer

gc.collect()

torch.cuda.empty_cache()

output_path = os.path.join(output_dir, "results.txt")

cache_dir = "./cache"
seed = 42
no_repeat_ngram_size = 3

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
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

results = []

generation_config = GenerationConfig(
    do_sample=True,
    temperature=temperature,
    num_beams=1,
    top_p=top_p,
    # top_k=top_k,
    no_repeat_ngram_size=no_repeat_ngram_size,
)

with open(output_path, "w", encoding="utf-8") as f:
    for (i, test_data) in enumerate(test_datas):
        predict = evaluate(test_data["instruction"], generation_config, max_len, test_data["input"], verbose=False)
        f.write(f"{i+1}. " + test_data["input"] + predict + "\n")
        print(f"{i+1}. " + test_data["input"] + predict)