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
import train_utils_pcba
import torch
import torch.optim as optim
import sys
import argparse
import copy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name')
    parser.add_argument('tokenizer')
    parser.add_argument('model')
    parser.add_argument('--lr', type=float, default=5e-7)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--bs', type=int, default=3)
    return parser.parse_args()

args = parse_args()


# %%
base_tk = Tokenizer.from_file(args.tokenizer)
tokenizer = BartTokenizerFast(tokenizer_object=base_tk)
tokenizer.backend_tokenizer.pre_tokinzer = PreTokenizer.custom(utils.SmilesPreTokenizer())


# %%
#NOTE: Datasets have had the last elements removed to make them even. TODO: Figure out why huggingface can't handle odd numbers.
train_raw = load_dataset('csv', data_files=['data/ogb_molpcba/train_pcba.csv'])
test_raw = load_dataset('csv', data_files=['data/ogb_molpcba/test_pcba.csv'])
valid_raw = load_dataset('csv', data_files=['data/ogb_molpcba/valid_pcba.csv'])


# %%
train_raw['train'][0]


# %%

classes = copy.deepcopy(train_raw.column_names['train'])
classes.remove('smiles')
classes.remove('mol_id')

print(len(classes))

assert len(classes) == 128, '128 classses not detected! Verify dataset correctness!'
def tokenize_function_pcba(examples):
    out_dict = tokenizer(examples['smiles'])
    labels = np.zeros((len(examples['smiles']), len(classes)))
    missing_mask = np.zeros((len(examples['smiles']), len(classes)))
    for i in range(len(examples['smiles'])):
        for j, prop in enumerate(classes):
            if examples[prop][i] is not None:
                missing_mask[i][j] = 1
                labels[i][j] = examples[prop][i]
            else:
                missing_mask[i][j] = 0
                labels[i][j] = -1

    out_dict['assay'] = labels
    out_dict['assay_missing'] = missing_mask
    return out_dict

train_ds = train_raw.map(tokenize_function_pcba, batched=True, remove_columns=train_raw.column_names['train'])['train']
test_ds = test_raw.map(tokenize_function_pcba, batched=True, remove_columns=train_raw.column_names['train'])['train']
valid_ds = valid_raw.map(tokenize_function_pcba, batched=True, remove_columns=train_raw.column_names['train'])['train']


# %%

model = DebertaV2ForSequenceClassification.from_pretrained(args.model,
                                                      config=args.model,
                                                      num_labels=len(classes),
                                                      pad_token_id=tokenizer.pad_token_id,
                                                    )

# %%
collator = DataCollatorWithPadding(tokenizer)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

batch_size = args.bs
# %%
train_utils_pcba.trainer(
    model=model,
    optimizer=optimizer,
    collator=collator,
    device=device,
    train_ds=train_ds,
    batch_size_train=batch_size,
    batch_size_eval=batch_size,
    num_epochs=15,
    model_save_dir=f"models/pcba/{args.experiment_name}",
    log_save_file=f"results/pcba/{args.experiment_name}.log",
    compute_metrics=True,
    eval_ds=test_ds,
    valid_ds=valid_ds,
    show_tqdm=True,
)
