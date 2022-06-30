"""Load tokenizer and save to local directory """

import os

from transformers import AutoTokenizer, AutoConfig


MODEL = "xlm-roberta-base"
local_dir = os.path.join("models", MODEL)
if not os.path.isdir(local_dir):
    os.mkdir(local_dir)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

tokenizer.save_pretrained(local_dir)
config.save_pretrained(local_dir)

# Test
# tokenizer = AutoTokenizer.from_pretrained(local_dir)
