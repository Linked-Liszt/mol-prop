from typing import List
import re
import csv

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