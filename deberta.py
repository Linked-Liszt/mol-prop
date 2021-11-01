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
import sys

tok_fp = sys.argv[1]
device_id = sys.argv[2]



# %%
base_tk = Tokenizer.from_file(tok_fp)
# Error with deberta tokenizer TODO: debug
tokenizer = BartTokenizerFast(tokenizer_object=base_tk)
tokenizer.backend_tokenizer.pre_tokinzer = PreTokenizer.custom(utils.SmilesPreTokenizer())


# %%
print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str('CS(=O)(=O)OCCCCOS(C)(=O)=O'))


# %%
print(tokenizer('CS(=O)(=O)OCCCCOS(C)(=O)=O'))
print(base_tk.encode('CS(=O)(=O)OCCCCOS(C)(=O)=O').ids)


# %%
train_raw = load_dataset('csv', data_files=['data/ogb_molhiv/train_hiv.csv'])
test_raw = load_dataset('csv', data_files=['data/ogb_molhiv/test_hiv.csv'])
valid_raw = load_dataset('csv', data_files=['data/ogb_molhiv/valid_hiv.csv'])


# %%
print(train_raw['train'][2]['smiles'])
input_ids =tokenizer(train_raw['train'][2]['smiles'])['input_ids']
print(input_ids)
print(tokenizer.convert_ids_to_tokens(input_ids))


# %%
def tokenize_function_hiv(examples):
    out_dict = tokenizer(examples["smiles"])
    return out_dict

train_ds = train_raw.map(tokenize_function_hiv, batched=True, remove_columns=["smiles","HIV_active", "mol_id"])['train']
test_ds = test_raw.map(tokenize_function_hiv, batched=True, remove_columns=["smiles","HIV_active", "mol_id"])['train']
valid_ds = valid_raw.map(tokenize_function_hiv, batched=True, remove_columns=["smiles","HIV_active", "mol_id"])['train']


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
n_layer = 24
model_config = DebertaV2Config(
    vocab_size=tokenizer.vocab_size,
    num_hidden_layers=n_layer,
    pad_token_id=tokenizer.pad_token_id,
)
model = DebertaV2ForMaskedLM(model_config)

data_collator = utils.PadPermCollator(tokenizer, DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True))
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-6)


# %%
batch_size = 3

if len(train_ds) % batch_size == 1:
    raise ValueError("Unable to handle batch size (train)")

if len(valid_ds) % batch_size == 1:
    raise ValueError("Unable to handle batch size (valid)")

train_utils.trainer(
    model=model,
    optimizer=optimizer,
    collator=data_collator,
    device=device,
    train_ds=train_ds,
    batch_size_train=batch_size,
    batch_size_eval=batch_size,
    num_epochs=50,
    model_save_dir=f"models/deberta-hiv-pre-{tok_fp.split('_')[1]}-l{n_layer}",
    log_save_file=f"results/deberta-hiv-pre-{tok_fp.split('_')[1]}-l{n_layer}.log",
    eval_ds=test_ds,
    compute_metrics=False,
    valid_ds=valid_ds
)
