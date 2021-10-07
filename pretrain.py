import os
import csv
from typing import List
import utils


def main():
    sms = import_smiles('data/HIV.csv', skiprow=True)

    for line in sms:
        print(line)


def import_smiles(data_fp: str, skiprow: bool = False) -> List[List[str]]:
    smiles_tokens = []
    with open(data_fp, 'r') as csv_f:
        if skiprow:
            csv_f.readline()
        raw_reader = csv.reader(csv_f, delimiter=',', quotechar='"')
        for row in raw_reader:
            if len(row) > 1:
                smiles_tokens.append(utils.tokenize_smiles(row[0]))
    return smiles_tokens


if __name__ == '__main__':
    main()