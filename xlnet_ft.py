import os
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')
from transformers import XLNetConfig, XLNetModel, XLNetTokenizer, XLNetLMHeadModel, XLNetForSequenceClassification
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import utils
from datasets import load_metric
from transformers.data.data_collator import DataCollatorWithPadding
import numpy as np
from tqdm import tqdm
import sklearn
import sys


#model_path = sys.argv[1]
model_path = 'abc'
lr = float(sys.argv[1])
save_path = sys.argv[2]

# %%
tokenizer = XLNetTokenizer(vocab_file='models/smiles_sp.model',
                           do_lower_case=False,
                           keep_accents=True
                           )


# %%
#NOTE: Datasets have had the last elements removed to make them even. TODO: Figure out why huggingface can't handle odd numbers.
train_raw = load_dataset('csv', data_files=['data/ogb_molhiv/train_hiv.csv'])
test_raw = load_dataset('csv', data_files=['data/ogb_molhiv/test_hiv.csv'])
valid_raw = load_dataset('csv', data_files=['data/ogb_molhiv/valid_hiv.csv'])


# %%

def tokenize_function_hiv(examples):
    out_dict = tokenizer(examples["smiles"])
    out_dict['label'] = [int(x) for x in examples['HIV_active']]
    return out_dict

train_ds = train_raw.map(tokenize_function_hiv, batched=True, remove_columns=["smiles","HIV_active", "mol_id"])['train']
test_ds = test_raw.map(tokenize_function_hiv, batched=True, remove_columns=["smiles","HIV_active", "mol_id"])['train']
valid_ds = valid_raw.map(tokenize_function_hiv, batched=True, remove_columns=["smiles","HIV_active", "mol_id"])['train']


if os.path.exists(model_path):
    model = XLNetForSequenceClassification.from_pretrained(model_path, config=model_path, num_labels=2)
else:
    model_config = XLNetConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=4,
        bi_data=True,
        num_labels=2
    )
    model = XLNetForSequenceClassification(model_config)

model.train()

# %%
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = metric.compute(predictions=predictions, references=labels)


    labels = np.asarray(labels)
    labels_oh = np.zeros((len(labels), max(labels)+1))
    labels_oh[np.arange(len(labels)),labels] = 1
    metrics['auc_roc'] = sklearn.metrics.roc_auc_score(labels_oh, logits)

    with open(save_path, 'a') as f:
        f.write(f"{metrics}\n")

    return metrics


# %%
training_args = TrainingArguments(
    f"models/xlnet-hiv",
    evaluation_strategy = "epoch",
    learning_rate=lr,
    weight_decay=0.01,
    per_device_train_batch_size=8, # has to be even?
    per_device_eval_batch_size=30,
    num_train_epochs=100,
    save_strategy='epoch',
)


# %%
collator = DataCollatorWithPadding(tokenizer)


# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    data_collator=collator,
)


# %%
print(trainer.evaluate())


# %%
trainer.train()
