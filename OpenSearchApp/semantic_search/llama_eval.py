import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model,PeftModel, get_peft_model_state_dict
from peft.tuners.lora import LoraLayer
from datasets import load_dataset,Dataset
import argparse
from utils import replicability
import bitsandbytes as bnb
import json
import os
import tqdm
import random

model = AutoModelForCausalLM.from_pretrained(
        "/home/ubuntu/checkpoint",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        #bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    )

model = PeftModel.from_pretrained(model, "/home/ubuntu/checkpoint")