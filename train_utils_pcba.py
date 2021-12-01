import torch
import sklearn
import numpy as np
from datasets import load_metric
from torch._C import Value
from tqdm import tqdm
import os
import torch.nn.functional as F
from contextlib import nullcontext
import copy
from ogb.graphproppred import Evaluator

metric = load_metric("accuracy")

def trainer(
    model,
    optimizer,
    collator,
    device,
    train_ds,
    batch_size_train,
    batch_size_eval,
    num_epochs,
    model_save_dir,
    log_save_file,
    compute_metrics,
    eval_ds = None,
    valid_ds = None,
    show_tqdm = False
):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if valid_ds is not None:
        metrics = evaluate(model, valid_ds, batch_size_eval, collator, device, compute_metrics, show_tqdm=show_tqdm)
        with open(log_save_file, 'a') as f:
            f.write(f"Valid Metrics: {metrics}\n")
        print(f"Valid Metrics: {metrics}")

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        train_loss = train(model, train_ds, batch_size_train, collator, device, optimizer, show_tqdm=show_tqdm)
        print(f"Train loss: {train_loss}")

        with open(log_save_file, 'a') as f:
            f.write(f"Epoch: {epoch}, Train Loss: {train_loss}\n")

        if eval_ds is not None:
            metrics = evaluate(model, eval_ds, batch_size_eval, collator, device, compute_metrics, show_tqdm=show_tqdm)
            with open(log_save_file, 'a') as f:
                f.write(f"Eval Metrics: {metrics}\n")
            print(f"Eval Metrics: {metrics}")

        if valid_ds is not None:
            metrics = evaluate(model, valid_ds, batch_size_eval, collator, device, compute_metrics, show_tqdm=show_tqdm)
            with open(log_save_file, 'a') as f:
                f.write(f"Valid Metrics: {metrics}\n")
            print(f"Valid Metrics: {metrics}")

        model.save_pretrained(os.path.join(model_save_dir, f"epoch_{epoch}"))



def train(model, dataset, batch_size, collator, device, optimizer, show_tqdm=False):
    model.train()
    losses = []

    dataset = dataset.shuffle()

    if show_tqdm:
        ctx = tqdm(total=len(dataset) // batch_size)
    else:
        ctx = nullcontext()

    with ctx as pbar:
        for i in range(0, len(dataset), batch_size):
            data = dataset[i: i + batch_size]
            processed_data = _multi_to(collator(data), device)
            prepped_data = {}
            prepped_data['input_ids'] = processed_data['input_ids']
            prepped_data['attention_mask'] = processed_data['attention_mask']

            optimizer.zero_grad()

            out = model(**prepped_data)

            loss = F.binary_cross_entropy_with_logits(out['logits'], processed_data['assay'], reduction='none')
            mean_loss = torch.mean(loss[processed_data['assay_missing'] == 1])

            if not torch.isnan(mean_loss):
                losses.append(mean_loss.detach().item())

                mean_loss.backward()
                optimizer.step()

            if show_tqdm:
                pbar.update(1)


    if len(dataset) % batch_size != 0:
        last_data = dataset[-(len(dataset) % batch_size):]
        processed_data = _multi_to(collator(last_data), device)
        prepped_data = {}
        prepped_data['input_ids'] = processed_data['input_ids']
        prepped_data['attention_mask'] = processed_data['attention_mask']

        optimizer.zero_grad()

        out = model(**prepped_data)

        loss = F.binary_cross_entropy_with_logits(out['logits'], processed_data['assay'], reduction='none')
        mean_loss = torch.mean(loss[processed_data['assay_missing'] == 1])

        if not torch.isnan(mean_loss):
            mean_loss.backward()
            optimizer.step()

    return torch.tensor(losses).mean().item()

def evaluate(model, dataset, batch_size, collator, device, compute_metrics, return_logits=False, show_tqdm=False):
    model.eval()
    if compute_metrics:
        logits = torch.tensor([]).to(device)
        labels = torch.tensor([]).to(device)
    losses = []

    if show_tqdm:
        ctx = tqdm(total=len(dataset) // batch_size)
    else:
        ctx = nullcontext()

    with ctx as pbar:
        for i in range(0, len(dataset), batch_size):
            data = dataset[i: i + batch_size]
            processed_data = _multi_to(collator(data), device)
            prepped_data = {}
            prepped_data['input_ids'] = processed_data['input_ids']
            prepped_data['attention_mask'] = processed_data['attention_mask']

            out = model(**prepped_data)
            if compute_metrics:
                logits = torch.cat((logits, out['logits'].detach()))
                labels = torch.cat((labels, processed_data['assay']))

            loss = F.binary_cross_entropy_with_logits(out['logits'], processed_data['assay'], reduction='none')
            mean_loss = torch.mean(loss[processed_data['assay_missing'] == 1])
            if not torch.isnan(mean_loss):
                losses.append(mean_loss.detach())
            if show_tqdm:
                pbar.update(1)


    if len(dataset) % batch_size != 0:
        last_data = dataset[-(len(dataset) % batch_size):]
        processed_data = _multi_to(collator(last_data), device)
        prepped_data = {}
        prepped_data['input_ids'] = processed_data['input_ids']
        prepped_data['attention_mask'] = processed_data['attention_mask']

        out = model(**prepped_data)
        if compute_metrics:
            logits = torch.cat((logits, out['logits'].detach()))
            labels = torch.cat((labels, processed_data['assay']))

        loss = F.binary_cross_entropy_with_logits(out['logits'], processed_data['assay'], reduction='none')
        mean_loss = torch.mean(loss[processed_data['assay_missing'] == 1])

        if not torch.isnan(mean_loss):
            losses.append(mean_loss.detach())

    if compute_metrics:
        out_logits = logits.cpu().numpy()
        out_labels = labels.cpu().numpy().astype(int)

        metrics = get_metrics(out_logits, out_labels)
    else:
        metrics = {}

    metrics['loss'] = torch.tensor(losses).mean().item()

    if return_logits:
        metrics['logits'] = out_logits
        metrics['labels'] = out_labels

    return metrics

def get_metrics(logits, labels):
    nan_labels = copy.deepcopy(labels).astype('float64')
    metrics = {}

    nan_labels[nan_labels==-1] = np.nan

    evaluator = Evaluator('ogbg-molpcba')
    metrics['ap'] = evaluator.eval({'y_true': nan_labels, 'y_pred': logits})['ap']
    return metrics


def _multi_to(data, device):
    if type(data) is dict:
        return {k: v.to(device) for k, v in data.items()}
    else:
        return data.to(device)