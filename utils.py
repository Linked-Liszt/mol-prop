from typing import List
import re
import csv
from tqdm import tqdm
import os

SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|\n#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

SMILES_REGEX = re.compile(SMI_REGEX_PATTERN)

def tokenize_smiles(text: str) -> List[str]:
    return [token for token in SMILES_REGEX.findall(text)]

def import_smiles(data_fp: str, skiprow: bool = False, include_y: bool = False):
    smiles_tokens = []
    ys = []
    with open(data_fp, 'r') as csv_f:
        if skiprow:
            csv_f.readline()
        raw_reader = csv.reader(csv_f, delimiter=',', quotechar='"')
        for row in raw_reader:
            if len(row) > 1:
                smiles_tokens.append(tokenize_smiles(row[0]))
                ys.append(int(row[2]))
    if include_y:
        return smiles_tokens, ys
    else:
        return smiles_tokens

def compute_max_length(dataset, data_key: str, tokenizer, max_len_path: str) -> int:
    if os.path.exists(max_len_path):
        max_len = int(open(max_len_path).read())
    else:
        max_len = 0
        with tqdm(total=len(dataset['train'])) as pbar:
            for data in dataset['train']:
                max_len = max(len(tokenizer(data[data_key])['input_ids']), max_len)
                pbar.update(1)
        with open(max_len_path, 'w') as f:
            f.write(str(max_len))

    # Nessesary for the data collator
    if max_len % 2 == 1:
        max_len += 1
    return max_len


class SmilesPreTokenizer:
    def smiles_split(self, i, normalized_string):
        tokens = tokenize_smiles(str(normalized_string))

        compiled_tokens = []
        char_idx = 0
        for token in tokens:
            compiled_tokens.append(normalized_string[char_idx:char_idx+len(token)])
            char_idx += len(token)
        return compiled_tokens

    def pre_tokenize(self, pretok):
        pretok.split(self.smiles_split)
