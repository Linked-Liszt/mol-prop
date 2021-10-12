import sentencepiece as spm
spm.SentencePieceTrainer.train(input='data/proc_zinc/all.txt',
                               input_sentence_size=5000000,
                               shuffle_input_sentence=True,
                               vocab_size=1000,
                               model_prefix='smiles_sp')