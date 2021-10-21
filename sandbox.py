# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# https://github.com/huggingface/notebooks/blob/master/examples/language_modeling_from_scratch.ipynb
# https://github.com/huggingface/transformers/tree/master/notebooks
# https://huggingface.co/transformers/model_doc/xlnet.html#transformers.XLNetTokenizer
# https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/pretrain_transformers_pytorch.ipynb#scrollTo=VE2MRZZhd5uM


# %%
import os
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')

from transformers import XLNetConfig, XLNetModel, XLNetTokenizer, XLNetTokenizerFast, XLNetLMHeadModel
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForPermutationLanguageModeling
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer
import utils
import train_utils


# %%
base_tk = Tokenizer.from_file("models/tk-vs1000_frozen.json")
tokenizer = XLNetTokenizerFast(tokenizer_object=base_tk)
tokenizer.backend_tokenizer.pre_tokinzer = PreTokenizer.custom(utils.SmilesPreTokenizer())


# %%
train_raw = load_dataset('csv', data_files=['data/ogb_molhiv/train_hiv.csv'])
test_raw = load_dataset('csv', data_files=['data/ogb_molhiv/test_hiv.csv'])
valid_raw = load_dataset('csv', data_files=['data/ogb_molhiv/valid_hiv.csv'])




print(tokenizer.pad_token_id)
print(tokenizer.unk_token_id)


# %%
def tokenize_function_hiv(examples):
    out_dict = tokenizer(examples["smiles"])
    return out_dict

train_ds = train_raw.map(tokenize_function_hiv, batched=True, remove_columns=["smiles","HIV_active", "mol_id"])['train']
test_ds = test_raw.map(tokenize_function_hiv, batched=True, remove_columns=["smiles","HIV_active", "mol_id"])['train']
valid_ds = valid_raw.map(tokenize_function_hiv, batched=True, remove_columns=["smiles","HIV_active", "mol_id"])['train']


# %%
data_collator = DataCollatorForPermutationLanguageModeling(tokenizer=tokenizer)


# %%
print(train_ds[0:3]['input_ids'])


# %%
class PadPermCollator():
    def __init__(self, tokenizer, collator):
        self.tokenizer = tokenizer
        self.collator = collator

    def __call__(self, data_list):
        max_len = -1
        for d in data_list['input_ids']:
            max_len = max(max_len, len(d))

        if max_len % 2 == 0:
            max_len += 1

        pad_data = self.tokenizer.pad(data_list,
                      padding='max_length',
                      max_length=max_len)

        return self.collator(pad_data)


# %%
test = PadPermCollator(tokenizer, data_collator)
print(test(train_ds[0:3]))
