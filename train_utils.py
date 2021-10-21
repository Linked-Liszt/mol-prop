import torch
import sklearn
import numpy as np
from datasets import load_metric
from torch._C import Value
from tqdm import tqdm
import os

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
    valid_ds = None
):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if valid_ds is not None:
        metrics = evaluate(model, valid_ds, batch_size_eval, collator, device, compute_metrics)
        with open(log_save_file, 'a') as f:
            f.write(f"Valid Metrics: {metrics}\n")
        print(f"Valid Metrics: {metrics}")

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        train_loss = train(model, train_ds, batch_size_train, collator, device, optimizer)
        print(f"Train loss: {train_loss}")

        with open(log_save_file, 'a') as f:
            f.write(f"Epoch: {epoch}, Train Loss: {train_loss}\n")

        if eval_ds is not None:
            metrics = evaluate(model, eval_ds, batch_size_eval, collator, device, compute_metrics)
            with open(log_save_file, 'a') as f:
                f.write(f"Eval Metrics: {metrics}\n")
            print(f"Eval Metrics: {metrics}")

        if valid_ds is not None:
            metrics = evaluate(model, valid_ds, batch_size_eval, collator, device, compute_metrics)
            with open(log_save_file, 'a') as f:
                f.write(f"Valid Metrics: {metrics}\n")
            print(f"Valid Metrics: {metrics}")

        model.save_pretrained(os.path.join(model_save_dir, f"epoch_{epoch}"))



def train(model, dataset, batch_size, collator, device, optimizer):
    if len(dataset) % batch_size == 1:
        raise ValueError("TODO: Unable to operate with rem 1. Change batch size.")

    model.train()
    loss = []

    with tqdm(total=len(dataset) // batch_size) as pbar:
        for i in range(0, len(dataset), batch_size):
            data = dataset[i: i + batch_size]
            prepped_data = _multi_to(collator(data), device)

            optimizer.zero_grad()

            out = model(**prepped_data)
            loss.append(out['loss'].detach().item())

            out['loss'].backward()
            optimizer.step()

            pbar.update(1)


    if len(dataset) % batch_size != 0:
        last_data = dataset[-(len(dataset) % batch_size):]
        prepped_data = _multi_to(collator(last_data), device)

        optimizer.zero_grad()

        out = model(**prepped_data)
        loss.append(out['loss'].detach().item())

        out['loss'].backward()
        optimizer.step()

    return torch.tensor(loss).mean().item()

def evaluate(model, dataset, batch_size, collator, device, compute_metrics):
    if len(dataset) % batch_size == 1:
        raise ValueError("TODO: Unable to operate with rem 1. Change batch size.")

    model.eval()
    if compute_metrics:
        logits = torch.tensor([]).to(device)
        labels = torch.tensor([]).to(device)
    loss = []

    with tqdm(total=len(dataset) // batch_size) as pbar:
        for i in range(0, len(dataset), batch_size):
            data = dataset[i: i + batch_size]
            prepped_data = _multi_to(collator(data), device)
            out = model(**prepped_data)
            if compute_metrics:
                logits = torch.cat((logits, out['logits'].detach()))
                labels = torch.cat((labels, prepped_data['labels']))
            loss.append(out['loss'].detach())
            pbar.update(1)


    if len(dataset) % batch_size != 0:
        last_data = dataset[-(len(dataset) % batch_size):]
        prepped_data = _multi_to(collator(last_data), device)
        out = model(**prepped_data)
        if compute_metrics:
            logits = torch.cat((logits, out['logits'].detach()))
            labels = torch.cat((labels, prepped_data['labels']))
        loss.append(out['loss'].detach())

    if compute_metrics:
        out_logits = logits.cpu().numpy()
        out_labels = labels.cpu().numpy().astype(int)

        metrics = get_metrics(out_logits, out_labels)
    else:
        metrics = {}

    metrics['loss'] = torch.tensor(loss).mean().item()

    return metrics

def get_metrics(logits, labels):
    predictions = np.argmax(logits, axis=-1)
    metrics = metric.compute(predictions=predictions, references=labels)

    labels_oh = np.zeros((len(labels), np.max(labels)+1))
    labels_oh[np.arange(len(labels)),labels] = 1
    metrics['auc_roc'] = sklearn.metrics.roc_auc_score(labels_oh, logits)
    return metrics


def _multi_to(data, device):
    if type(data) is dict:
        return {k: v.to(device) for k, v in data.items()}
    else:
        return data.to(device)