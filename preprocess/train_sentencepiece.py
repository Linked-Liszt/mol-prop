# SPM References: https://github.com/google/sentencepiece/blob/master/python/README.md
# https://github.com/google/sentencepiece/blob/master/doc/options.md


import sentencepiece as spm
spm.SentencePieceTrainer.train(input='data/ogb_molhiv/hiv_raw.txt',
                               input_sentence_size=7000000,
                               shuffle_input_sentence=True,
                               vocab_size=1092,
                               model_prefix='models/smiles_hiv_sp')