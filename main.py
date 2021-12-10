import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import DECODE
from fairseq.data.data_utils import collate_tokens
from functools import partial
from tqdm import tqdm
import numpy as np
import pickle


def collate_batch(batch, roberta, unstructured=False):
    if unstructured:
        pass
    else:
        utterances_batch, labels_batch = [], []
        for _utterances_pairs, _labels in batch:
            utterances_batch.append(roberta.encode(*_utterances_pairs))
            labels_batch.append(_labels)

        utterances_batch = collate_tokens(utterances_batch, pad_idx=1)
        labels_batch = torch.tensor(labels_batch, dtype=torch.int64)

        return utterances_batch, labels_batch


def train(model, data_loader, loss_fn, optimizer):
    losses = []
    model.train()
    for utterances, labels in tqdm(data_loader):
        utterances = utterances.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        labels_pred = roberta.predict('contradiction_detect', utterances)
        loss = loss_fn(labels_pred, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def validate(model, data_loader, loss_fn):
    losses = []
    correct = 0
    with torch.no_grad():
        model.eval()
        for utterances, labels in tqdm(data_loader):
            utterances = utterances.cuda()
            labels = labels.cuda()
            labels_pred = roberta.predict('contradiction_detect', utterances)
            correct += torch.sum(torch.argmax(labels_pred, dim=1) == labels).item()
            loss = loss_fn(labels_pred, labels)
            losses.append(loss.item())
    return np.mean(losses), correct / len(data_loader.dataset)


if __name__ == "__main__":
    EPOCHS = 20
    BATCH_SIZE = 32

    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    # roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
    roberta.register_classification_head('contradiction_detect', num_classes=2)
    roberta.eval()

    unstructured = False
    train_data = DECODE("decode_v0.1/train.jsonl", unstructured)
    val_data = DECODE("decode_v0.1/dev.jsonl", unstructured)
    test_data = DECODE("decode_v0.1/test.jsonl", unstructured)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=partial(collate_batch, roberta=roberta, unstructured=unstructured))
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True,
                                collate_fn=partial(collate_batch, roberta=roberta, unstructured=unstructured))
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True,
                                 collate_fn=partial(collate_batch, roberta=roberta, unstructured=unstructured))

    roberta.cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(roberta.parameters(), lr=1e-5, weight_decay=0.1)

    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch + 1}")
        train_loss = train(roberta, train_dataloader, loss_fn, optimizer)
        print(f"    Train loss: {train_loss}")
        train_losses.append(train_loss)
        torch.save(roberta.model.state_dict(), f"checkpoint_{epoch + 1}.pt")

        val_loss, val_acc = validate(roberta, val_dataloader, loss_fn)
        print(f"    Validation loss: {val_loss}")
        print(f"    Validation accuracy: {val_acc}")
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    test_loss, test_acc = validate(roberta, test_dataloader, loss_fn)
    print("*" * 32)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_acc}")

    results = (train_losses, val_losses, val_accs, test_loss, test_acc)
    with open('results.pickle', 'wb') as f:
        pickle.dump(results, f)
