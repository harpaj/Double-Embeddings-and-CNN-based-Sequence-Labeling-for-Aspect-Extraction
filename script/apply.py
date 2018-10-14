import argparse
from collections import Counter
import json
from os import path

import numpy as np
import torch
import torch.nn as tnn

np.random.seed(1337)
torch.manual_seed(1337)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../data"
PREPARED_DIR = "prepared"
FN_DATASET = "dataset_{language}.npz"
FN_WORD_IDX = "word_idx_{language}.json"
MODEL_DIR = "models"
FN_RESULT = "xu_aspect_terms_{model_name}.tsv"
EMBEDDING_DIR = "embeddings"
FN_GEN_EMBEDDING = "gen_{language}.vec.npy"
FN_RESTO_EMBEDDING = "restaurant_{language}.vec.npy"


def get_aspect_terms(X, y):
    current_term = []
    for row_idx, y_row in enumerate(y):
        for col_idx, y_col in enumerate(y_row):
            if current_term:
                if y_col in (0, 1):
                    yield current_term
                    current_term = []
            if y_col in (1, 2):
                x_col = X[row_idx][col_idx]
                if x_col not in (0, 1, 2):
                    current_term.append(x_col)
        if current_term:
            yield current_term
            current_term = []


def apply(language, model_name, batch_size=128):
    model = torch.load(path.join(DATA_DIR, language, MODEL_DIR, "xu" + model_name))
    resto_embedding = np.load(path.join(DATA_DIR, language, EMBEDDING_DIR, FN_RESTO_EMBEDDING.format(language=language)))
    resto_embedding = torch.from_numpy(resto_embedding).to(device)
    model.domain_embedding = tnn.Embedding(resto_embedding.shape[0], resto_embedding.shape[1], padding_idx=0)
    model.domain_embedding.weight = tnn.Parameter(resto_embedding, requires_grad=False)
    gen_embedding = np.load(path.join(DATA_DIR, language, EMBEDDING_DIR, FN_GEN_EMBEDDING.format(language=language)))
    gen_embedding = torch.from_numpy(gen_embedding).to(device)
    model.gen_embedding = tnn.Embedding(gen_embedding.shape[0], gen_embedding.shape[1], padding_idx=0)
    model.gen_embedding.weight = tnn.Parameter(gen_embedding, requires_grad=False)
    model.eval()

    dataset = np.load(path.join(DATA_DIR, language, PREPARED_DIR, FN_DATASET.format(language=language)))["train_X"]

    aspect_terms = Counter()

    for offset in range(0, dataset.shape[0], batch_size):
        batch_test_X_len = np.sum(dataset[offset:offset+batch_size] != 0, axis=1)
        batch_idx = batch_test_X_len.argsort()[::-1]
        batch_test_X_len = batch_test_X_len[batch_idx]
        batch_test_X_mask = (dataset[offset:offset+batch_size] != 0)[batch_idx].astype(np.uint8)
        batch_test_X = dataset[offset:offset+batch_size][batch_idx]
        batch_test_X_mask = torch.from_numpy(batch_test_X_mask).long().to(device)
        batch_test_X = torch.from_numpy(batch_test_X).long().to(device)
        batch_pred_y = model(batch_test_X, batch_test_X_len, batch_test_X_mask, testing=True)
        batch_pred_y = batch_pred_y.data.to("cpu").numpy().argmax(axis=2)

        for aspect_term in get_aspect_terms(batch_test_X.to("cpu").numpy(), batch_pred_y):
            aspect_terms[tuple(aspect_term)] += 1

    with open(path.join(DATA_DIR, language, PREPARED_DIR, FN_WORD_IDX.format(language=language))) as fh:
        word_idx = json.load(fh)

    idx_word = {v: k for k, v in word_idx.items()}

    aspect_term_words = Counter()
    for aspect_term, freq in aspect_terms.most_common():
        aspect_term_words[" ".join(idx_word[i].lower() for i in aspect_term)] += freq

    with open(path.join(DATA_DIR, language, MODEL_DIR, FN_RESULT.format(model_name=model_name)), 'w') as fh:
        for aspect_term, freq in aspect_term_words.most_common():
            fh.write(aspect_term + "\t" + str(freq) + "\n")
        print(f"Extracted and wrote {len(aspect_term_words)} aspect terms!")


parser = argparse.ArgumentParser()
parser.add_argument('--language', type=str, default="finnish")
parser.add_argument('--model_name', type=str, required=True)
args = parser.parse_args()

apply(args.language, args.model_name)
