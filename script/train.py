import argparse
import torch
import numpy as np
import random
import seqeval.metrics

from model import Model

np.random.seed(15476)
random.seed(15476)
torch.manual_seed(15476)
# torch.cuda.manual_seed(1337)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

idx_to_tag = {0: "O", 1: "B-ASPECT", 2: "I-ASPECT"}.get

def batch_generator(X, y, batch_size=128, return_idx=False):
    for offset in range(0, X.shape[0], batch_size):
        batch_X_len = np.sum(X[offset:offset+batch_size] != 0, axis=1)
        batch_y_len = np.sum(y[offset:offset+batch_size] != -1, axis=1)
        assert np.array_equal(batch_X_len, batch_y_len)
        batch_idx = batch_X_len.argsort()[::-1]
        batch_X_len = batch_X_len[batch_idx]
        batch_X_mask = (X[offset:offset+batch_size] != 0)[batch_idx].astype(np.uint8)
        batch_X = X[offset:offset+batch_size][batch_idx]
        batch_y = y[offset:offset+batch_size][batch_idx]
        batch_X = torch.autograd.Variable(torch.from_numpy(batch_X).long().to(device))
        batch_X_mask = torch.autograd.Variable(torch.from_numpy(batch_X_mask).long().to(device))
        batch_y = torch.autograd.Variable(torch.from_numpy(batch_y).long().to(device))
        if len(batch_y.size()) == 2:
            batch_y = torch.nn.utils.rnn.pack_padded_sequence(batch_y, batch_X_len, batch_first=True)
        if return_idx:  # in testing, need to sort back.
            yield (batch_X, batch_y, batch_X_len, batch_X_mask, batch_idx)
        else:
            yield (batch_X, batch_y, batch_X_len, batch_X_mask)


def valid_performance(model, valid_X, valid_y):
    model.eval()
    for batch in batch_generator(valid_X, valid_y, batch_size=valid_X.shape[0], return_idx=True):
        batch_valid_X, batch_valid_y, batch_valid_X_len, batch_valid_X_mask, batch_idx = batch
        predicted = model(batch_valid_X, batch_valid_X_len, batch_valid_X_mask, batch_valid_y, testing=True)
        predicted = predicted.data.cpu().numpy().argmax(axis=2)
        valid_y = valid_y[batch_idx]
        truths = []
        predictions = []
        for idx, length in enumerate(batch_valid_X_len):
            truths.append([idx_to_tag(e) for e in valid_y[idx][:length]])
            predictions.append([idx_to_tag(e) for e in predicted[idx][:length]])
        model.train()
        return seqeval.metrics.classification_report(truths, predictions, digits=3)


def valid_loss(model, valid_X, valid_y):
    model.eval()
    losses = []
    for batch in batch_generator(valid_X, valid_y):
        batch_valid_X, batch_valid_y, batch_valid_X_len, batch_valid_X_mask = batch
        loss = model(batch_valid_X, batch_valid_X_len, batch_valid_X_mask, batch_valid_y)
        losses.append(loss.data[0])
    model.train()
    return sum(losses)/len(losses)


def train(train_X, train_y, valid_X, valid_y, model, model_fn, optimizer, parameters, epochs=200, batch_size=128):
    best_loss = float("inf")
    best_state = None
    valid_history = []
    train_history = []
    for epoch in range(epochs):
        print("Epoch", epoch)
        for batch in batch_generator(train_X, train_y, batch_size):
            batch_train_X, batch_train_y, batch_train_X_len, batch_train_X_mask = batch
            # print(batch_train_X)
            # print(batch_train_y)
            loss = model(batch_train_X, batch_train_X_len, batch_train_X_mask, batch_train_y)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, 1.)
            optimizer.step()
        loss = valid_loss(model, train_X, train_y)
        print("Train Loss", loss)
        train_history.append(loss)
        # print(loss)
        # print(valid_performance(model, train_X, train_y))
        loss = valid_loss(model, valid_X, valid_y)
        print("Valid Loss", loss)
        valid_history.append(loss)
        # print(loss)
        # print(valid_performance(model, valid_X, valid_y))
        if loss < best_loss:
            best_loss = loss
            best_state = model.state_dict()
        shuffle_idx = np.random.permutation(len(train_X))
        train_X = train_X[shuffle_idx]
        train_y = train_y[shuffle_idx]

    model.load_state_dict(best_state)

    print("=== EVAL RESULTS ===")
    print("Report for", model_fn)
    print(valid_performance(model, valid_X, valid_y))

    torch.save(model, model_fn)
    return train_history, valid_history


def run(domain, data_dir, model_dir, valid_split, runs, epochs, lr, dropout, model_name="", batch_size=128):
    gen_emb = np.load(data_dir+"embeddings/gen_english.vec.npy")
    domain_emb = np.load(data_dir+"embeddings/restaurant_english.vec.npy")
    ae_data = np.load(data_dir+"prepared/dataset_english_annotated.npz")

    valid_X = ae_data['train_X'][-valid_split:]
    valid_y = ae_data['train_y'][-valid_split:]
    train_X = ae_data['train_X'][:-valid_split]
    train_y = ae_data['train_y'][:-valid_split]

    for r in range(runs):
        print(r)
        model = Model(gen_emb, domain_emb, 3, dropout=dropout)
        model.to(device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=lr)
        train_history, valid_history = train(
            train_X, train_y, valid_X, valid_y, model, model_dir+"xu"+model_name+"_"+str(r),
            optimizer, parameters, epochs, batch_size
        )


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default="../data/english/models/")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--domain', type=str, default="restaurant")
parser.add_argument('--data_dir', type=str, default="../data/english/")
parser.add_argument('--valid', type=int, default=150)  # number of validation data.
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--dropout', type=float, default=0.55)
parser.add_argument('--model_name', type=str, default="")

args = parser.parse_args()

run(args.domain, args.data_dir, args.model_dir, args.valid, args.runs, args.epochs, args.lr, args.dropout, args.model_name, args.batch_size)
