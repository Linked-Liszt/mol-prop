# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os

from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')
from transformers import BartTokenizerFast, DebertaV2ForSequenceClassification
from datasets import load_dataset
import utils
from transformers.data.data_collator import DataCollatorWithPadding
import numpy as np
from tqdm import tqdm
import utils
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer
import train_utils
import torch
import torch.optim as optim
import sys
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name')
    parser.add_argument('tokenizer')
    parser.add_argument('model')
    parser.add_argument('--lr', type=float, default=5e-7)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--bs', type=int, default=1)
    return parser.parse_args()

args = parse_args()


# %%
base_tk = Tokenizer.from_file(args.tokenizer)
tokenizer = BartTokenizerFast(tokenizer_object=base_tk)
tokenizer.backend_tokenizer.pre_tokinzer = PreTokenizer.custom(utils.SmilesPreTokenizer())


# %%
#NOTE: Datasets have had the last elements removed to make them even. TODO: Figure out why huggingface can't handle odd numbers.
train_raw = load_dataset('csv', data_files=['data/ogb_molhiv/train_hiv.csv'])
test_raw = load_dataset('csv', data_files=['data/ogb_molhiv/test_hiv.csv'])
valid_raw = load_dataset('csv', data_files=['data/ogb_molhiv/valid_hiv.csv'])


# %%
train_raw['train'][0]


# %%

def tokenize_function_hiv(examples):
    out_dict = tokenizer(examples['smiles'])

    out_dict['label'] = [int(x) for x in examples['HIV_active']]
    return out_dict

train_ds = train_raw.map(tokenize_function_hiv, batched=True, remove_columns=["smiles","HIV_active", "mol_id"])['train']
test_ds = test_raw.map(tokenize_function_hiv, batched=True, remove_columns=["smiles","HIV_active", "mol_id"])['train']
valid_ds = valid_raw.map(tokenize_function_hiv, batched=True, remove_columns=["smiles","HIV_active", "mol_id"])['train']


# %%

model = DebertaV2ForSequenceClassification.from_pretrained(args.model,
                                                      config=args.model,
                                                      num_labels=2,
                                                      pad_token_id=tokenizer.pad_token_id,
                                                    )

"""
n_layer = 24
model_config = DebertaV2Config(
    vocab_size=tokenizer.vocab_size,
    num_hidden_layers=n_layer,
    num_labels=2,
    pad_token_id=tokenizer.pad_token_id,
)
model =  DebertaV2ForSequenceClassification(model_config)
"""


# %%
collator = DataCollatorWithPadding(tokenizer)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

batch_size = args.bs
# %%
train_utils.trainer(
    model=model,
    optimizer=optimizer,
    collator=collator,
    device=device,
    train_ds=train_ds,
    batch_size_train=batch_size,
    batch_size_eval=batch_size,
    num_epochs=20,
    model_save_dir=f"models/{args.experiment_name}",
    log_save_file=f"results/{args.experiment_name}.log",
    compute_metrics=True,
    eval_ds=test_ds,
    valid_ds=valid_ds
)
