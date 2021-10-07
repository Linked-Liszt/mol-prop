from typing import List
import re

SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|\n#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

SMILES_REGEX = re.compile(SMI_REGEX_PATTERN)

def tokenize_smiles(text: str) -> List[str]:
    return [token for token in SMILES_REGEX.findall(text)]