# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from tokenizers import SentencePieceBPETokenizer
import utils
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import PreTokenizer
from transformers import XLNetTokenizer
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer


# %%
with open('data/ogb_molhiv/hiv.csv', 'r') as hiv_f, open('data/ogb_molhiv/hiv_raw.txt', 'w') as hiv_raw:
    next(hiv_f)
    for line in hiv_f:
        hiv_raw.write(line.split(',')[1] + '\n')


# %%
class SmilesPreTokenizer:
    def smiles_split(self, i, normalized_string):
        tokens = utils.tokenize_smiles(str(normalized_string))

        compiled_tokens = []
        char_idx = 0
        for token in tokens:
            compiled_tokens.append(normalized_string[char_idx:char_idx+len(token)])
            char_idx += len(token)
        return compiled_tokens

    def pre_tokenize(self, pretok):
        pretok.split(self.smiles_split)


# %%
tok = Tokenizer(BPE())
pre_tokenizer = PreTokenizer.custom(SmilesPreTokenizer())
print(pre_tokenizer.pre_tokenize_str('CS(=O)(=O)OCCCCOS(C)(=O)=O'))


# %%
special_tokens = ["<unk>", "<sep>", "<pad>", "<cls>", "<mask", "<eop>", "<eod>"]
xlnet_tokenizer = Tokenizer(BPE())
xlnet_tokenizer.pre_tokenizer = Split(pattern=utils.SMI_REGEX_PATTERN, behavior='contiguous')
trainer = BpeTrainer(special_tokens=special_tokens, show_progress=True)
files = ['data/ogb_molhiv/hiv_raw.txt']


# %%
xlnet_tokenizer.train(files, trainer)
xlnet_tokenizer.save("models/xlnet_custom.json")
