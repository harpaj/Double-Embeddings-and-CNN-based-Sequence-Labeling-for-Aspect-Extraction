import csv
import nltk.data
from nltk.tokenize import wordpunct_tokenize
import itertools as it
import numpy as np
import json

DATA_DIR = "data/"
LABELED = "reviews_fi_tagged_100.tsv"
OUT_INDEX = "word_idx_fi.json"
OUT_DATA = "restaurant_fi.npz"
sent_detector = nltk.data.load('tokenizers/punkt/finnish.pickle')


def tokenize_with_tag(string, label):
    token_tags = []
    for token in wordpunct_tokenize(string):
        token_tags.append([token, label])
    return token_tags


def get_token_tags(sent):
    lastpos = 0
    token_tags = []
    while True:
        start = sent.find("<target>", lastpos)
        if start > -1:
            token_tags += tokenize_with_tag(sent[lastpos:start], "O")
            sent = sent.replace("<target>", "", 1)
            end = sent.find("</target>", lastpos)
            if not end:
                raise IndexError("S {}: One end to little".format(sent))
            i_tags = tokenize_with_tag(sent[start:end], "I")
            i_tags[0][1] = "B"
            token_tags += i_tags
            sent = sent.replace("</target>", "", 1)
            lastpos = end
        else:
            break
    token_tags += tokenize_with_tag(sent[lastpos:], "O")
    try:
        assert "<target>" not in sent
        assert "</target>" not in sent
    except AssertionError:
        raise AssertionError("S {}: tags are left".format(sent))
    return token_tags


def map_and_split(all_token_tags, train_ratio=0.8):
    index = {}
    keyiter = it.count()
    all_tokens = []
    all_tags = []
    maxlen = len(max(all_token_tags, key=len))
    for tt in all_token_tags:
        tokens = [0] * maxlen
        tags = [0] * maxlen
        for pos, (token, tag) in enumerate(tt):
            key = index.setdefault(token, next(keyiter))
            tokens[pos] = key
            tags[pos] = tag
        all_tokens.append(tokens)
        all_tags.append(tags)
    split = int(len(all_tokens) * train_ratio)
    data = {
        "train_X": np.array(all_tokens[:split]),
        "test_X": np.array(all_tokens[split:]),
        "train_y": np.array(all_tags[:split]),
        "test_y": np.array(all_tags[split:])
    }
    return index, data


with open(DATA_DIR + LABELED, 'r') as infh:
    reader = csv.reader(infh, delimiter='\t')
    all_token_tags = []
    for row in reader:
        _, review, _, _ = row
        if review[0] == review[-1] == '"':
            review = review[1:-1]
        review = review.replace('""', '"')
        for sent in sent_detector.tokenize(review):
            all_token_tags.append(get_token_tags(sent))
    index, data = map_and_split(all_token_tags)
    np.savez_compressed(DATA_DIR + OUT_DATA, **data)
    with open(DATA_DIR + OUT_INDEX, 'w') as outfh:
        json.dump(index, outfh, ensure_ascii=False)
