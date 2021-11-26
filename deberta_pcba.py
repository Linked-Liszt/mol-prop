# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# https://github.com/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch.ipynb
# https://github.com/huggingface/transformers/tree/master/notebooks
# https://huggingface.co/transformers/model_doc/xlnet.html#transformers.XLNetTokenizer
# https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/pretrain_transformers_pytorch.ipynb#scrollTo=VE2MRZZhd5uM


# %%
import os
from datasets.features.features import Value
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')

from transformers import DebertaTokenizerFast, DebertaV2ForMaskedLM, DebertaV2Config, BartTokenizerFast
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from tokenizers import Tokenizer
from tqdm import tqdm
from tokenizers.pre_tokenizers import PreTokenizer
import utils
import torch
import torch.optim as optim
import train_utils
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name')
    parser.add_argument('tokenizer')
    parser.add_argument('--lr', type=float, default=5e-7)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--bs', type=int, default=3)
    return parser.parse_args()

args = parse_args()

# %%
base_tk = Tokenizer.from_file(args.tokenizer)
# Error with deberta tokenizer TODO: debug
tokenizer = BartTokenizerFast(tokenizer_object=base_tk)
tokenizer.backend_tokenizer.pre_tokinzer = PreTokenizer.custom(utils.SmilesPreTokenizer())


# %%
print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str('CS(=O)(=O)OCCCCOS(C)(=O)=O'))


# %%
print(tokenizer('CS(=O)(=O)OCCCCOS(C)(=O)=O'))
print(base_tk.encode('CS(=O)(=O)OCCCCOS(C)(=O)=O').ids)


# %%
train_raw = load_dataset('csv', data_files=['data/ogb_molpcba/train_pcba.csv'])
test_raw = load_dataset('csv', data_files=['data/ogb_molpcba/test_pcba.csv'])
valid_raw = load_dataset('csv', data_files=['data/ogb_molpcba/valid_pcba.csv'])

# %%
print(train_raw['train'][2]['smiles'])
input_ids =tokenizer(train_raw['train'][2]['smiles'])['input_ids']
print(input_ids)
print(tokenizer.convert_ids_to_tokens(input_ids))


# %%
def tokenize_function_hiv(examples):
    out_dict = tokenizer(examples["smiles"])
    return out_dict

train_ds = train_raw.map(tokenize_function_hiv, batched=True, remove_columns=train_raw.column_names['train'])['train']
test_ds = test_raw.map(tokenize_function_hiv, batched=True, remove_columns=train_raw.column_names['train'])['train']
valid_ds = valid_raw.map(tokenize_function_hiv, batched=True, remove_columns=train_raw.column_names['train'])['train']


# %%
"""
model_path = 'models/bart-pre-hiv-b3-l4/checkpoint-208373'
model = BartForConditionalGeneration.from_pretrained(model_path,
                                                      config=model_path,
                                                      pad_token_id=tokenizer.pad_token_id,
                                                      bos_token_id=tokenizer.bos_token_id,
                                                      eos_token_id=tokenizer.eos_token_id,
)
"""
n_layer = 9
model_config = DebertaV2Config(
    vocab_size=tokenizer.vocab_size,
    num_hidden_layers=n_layer,
    pad_token_id=tokenizer.pad_token_id,
    max_position_embeddings=1024
)
model = DebertaV2ForMaskedLM(model_config)

data_collator = utils.PadPermCollator(tokenizer, DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True))
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-6)


# %%
batch_size = args.bs

train_utils.trainer(
    model=model,
    optimizer=optimizer,
    collator=data_collator,
    device=device,
    train_ds=train_ds,
    batch_size_train=batch_size,
    batch_size_eval=batch_size,
    num_epochs=100,
    model_save_dir=f"models/pcba/deberta-pcba-pre-{args.experiment_name}-l{n_layer}",
    log_save_file=f"results/pcba/eberta-pcba-pre-{args.experiment_name}-l{n_layer}",
    eval_ds=test_ds,
    compute_metrics=False,
    valid_ds=valid_ds
)
