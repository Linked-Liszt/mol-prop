import csv
from tokenizers import ByteLevelBPETokenizer
from typing import List


def main():
    sms = import_smiles('data/HIV.csv', skiprow=True)
    tokens = []
    for mol in sms:
        for char in mol:
            if char not in tokens:
                tokens.append(char)

    print(len(tokens))
    for tok in sms:
        print(tok)


def import_smiles(data_fp: str, skiprow: bool = False) -> List[str]:
    smiles_strings = []
    with open(data_fp, 'r') as csv_f:
        if skiprow:
            csv_f.readline()
        raw_reader = csv.reader(csv_f, delimiter=',', quotechar='"')
        for row in raw_reader:
            if len(row) > 1:
                smiles_strings.append(row[0])
    return smiles_strings


if __name__ == '__main__':
    main()