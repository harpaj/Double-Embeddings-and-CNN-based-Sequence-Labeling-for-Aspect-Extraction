import argparse
import torch
import json
import numpy as np
import random
import xml.etree.ElementTree as ET
from subprocess import check_output

from model import Model

np.random.seed(1337)
random.seed(1337)
torch.manual_seed(1337)
# torch.cuda.manual_seed(1337)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def label_rest_xml(fn, output_fn, corpus, label):
    print(output_fn)
    dom = ET.parse(fn)
    root = dom.getroot()
    pred_y = []
    for zx, sent in enumerate(root.iter("sentence")):
        tokens = corpus[zx]
        lb = label[zx]
        opins = ET.Element("Opinions")
        token_idx, pt, tag_on = 0, 0, False
        start, end = -1, -1
        for ix, c in enumerate(sent.find('text').text):
            if token_idx < len(tokens) and pt >= len(tokens[token_idx]):
                pt = 0
                token_idx += 1

            if token_idx < len(tokens) and lb[token_idx] == 1 and pt == 0 and c != ' ':
                if tag_on:
                    end = ix
                    tag_on = False
                    opin = ET.Element("Opinion")
                    opin.attrib['target'] = sent.find('text').text[start:end]
                    opin.attrib['from'] = str(start)
                    opin.attrib['to'] = str(end)
                    opins.append(opin)
                start = ix
                tag_on = True
            elif token_idx < len(tokens) and lb[token_idx] == 2 and pt == 0 and c != ' ' and not tag_on:
                start = ix
                tag_on = True
            elif token_idx < len(tokens) and (lb[token_idx] == 0 or lb[token_idx] == 1) and tag_on and pt == 0:
                end = ix
                tag_on = False
                opin = ET.Element("Opinion")
                opin.attrib['target'] = sent.find('text').text[start:end]
                opin.attrib['from'] = str(start)
                opin.attrib['to'] = str(end)
                opins.append(opin)
            elif token_idx >= len(tokens) and tag_on:
                end = ix
                tag_on = False
                opin = ET.Element("Opinion")
                opin.attrib['target'] = sent.find('text').text[start:end]
                opin.attrib['from'] = str(start)
                opin.attrib['to'] = str(end)
                opins.append(opin)
            if c == ' ':
                pass
            elif tokens[token_idx][pt:pt+2] == '``' or tokens[token_idx][pt:pt+2] == "''":
                pt += 2
            else:
                pt += 1
        if tag_on:
            tag_on = False
            end = len(sent.find('text').text)
            opin = ET.Element("Opinion")
            opin.attrib['target'] = sent.find('text').text[start:end]
            opin.attrib['from'] = str(start)
            opin.attrib['to'] = str(end)
            opins.append(opin)
        sent.append(opins)
    dom.write(output_fn)


def label_laptop_xml(fn, output_fn, corpus, label):
    dom = ET.parse(fn)
    root = dom.getroot()
    pred_y = []
    for zx, sent in enumerate(root.iter("sentence")):
        tokens = corpus[zx]
        lb = label[zx]
        opins = ET.Element("aspectTerms")
        token_idx, pt, tag_on = 0, 0, False
        start, end = -1, -1
        for ix, c in enumerate(sent.find('text').text):
            if token_idx < len(tokens) and pt >= len(tokens[token_idx]):
                pt = 0
                token_idx += 1

            if token_idx < len(tokens) and lb[token_idx] == 1 and pt == 0 and c != ' ':
                if tag_on:
                    end = ix
                    tag_on = False
                    opin = ET.Element("aspectTerm")
                    opin.attrib['term'] = sent.find('text').text[start:end]
                    opin.attrib['from'] = str(start)
                    opin.attrib['to'] = str(end)
                    opins.append(opin)
                start = ix
                tag_on = True
            elif token_idx < len(tokens) and lb[token_idx] == 2 and pt == 0 and c != ' ' and not tag_on:
                start = ix
                tag_on = True
            elif token_idx < len(tokens) and (lb[token_idx] == 0 or lb[token_idx] == 1) and tag_on and pt == 0:
                end = ix
                tag_on = False
                opin = ET.Element("aspectTerm")
                opin.attrib['term'] = sent.find('text').text[start:end]
                opin.attrib['from'] = str(start)
                opin.attrib['to'] = str(end)
                opins.append(opin)
            elif token_idx >= len(tokens) and tag_on:
                end = ix
                tag_on = False
                opin = ET.Element("aspectTerm")
                opin.attrib['term'] = sent.find('text').text[start:end]
                opin.attrib['from'] = str(start)
                opin.attrib['to'] = str(end)
                opins.append(opin)
            if c == ' ' or ord(c) == 160:
                pass
            elif tokens[token_idx][pt:pt+2] == '``' or tokens[token_idx][pt:pt+2] == "''":
                pt += 2
            else:
                pt += 1
        if tag_on:
            tag_on = False
            end = len(sent.find('text').text)
            opin = ET.Element("aspectTerm")
            opin.attrib['term'] = sent.find('text').text[start:end]
            opin.attrib['from'] = str(start)
            opin.attrib['to'] = str(end)
            opins.append(opin)
        sent.append(opins)
    dom.write(output_fn)


def test(model, test_X, raw_X, domain, command, template, batch_size=128, crf=False):
    pred_y = np.zeros((test_X.shape[0], 83), np.int16)
    model.eval()
    for offset in range(0, test_X.shape[0], batch_size):
        batch_test_X_len = np.sum(test_X[offset:offset+batch_size] != 0, axis=1)
        batch_idx = batch_test_X_len.argsort()[::-1]
        batch_test_X_len = batch_test_X_len[batch_idx]
        print(batch_test_X_len[0])
        batch_test_X_mask = (test_X[offset:offset+batch_size] != 0)[batch_idx].astype(np.uint8)
        batch_test_X = test_X[offset:offset+batch_size][batch_idx]
        # print(batch_test_X)
        batch_test_X_mask = torch.autograd.Variable(torch.from_numpy(batch_test_X_mask).long().to(device))
        batch_test_X = torch.autograd.Variable(torch.from_numpy(batch_test_X).long().to(device))
        batch_pred_y = model(batch_test_X, batch_test_X_len, batch_test_X_mask, testing=True)
        r_idx = batch_idx.argsort()
        batch_pred_y = batch_pred_y.data.to("cpu").numpy().argmax(axis=2)[r_idx]
        # print(batch_pred_y)
        pred_y[offset:offset+batch_size, :batch_pred_y.shape[1]] = batch_pred_y
    # model.train()
    assert len(pred_y) == len(test_X)

    command = command.split()
    if domain == 'restaurant':
        label_rest_xml(template, command[8], raw_X, pred_y)
        acc = check_output(command).split()
        print(acc)
        return float(acc[9][10:])
    elif domain == 'laptop':
        label_laptop_xml(template, command[4], raw_X, pred_y)
        acc = check_output(command).split()
        print(acc)
        return float(acc[15])


def recreate_data(data_dir, test_X):
    with open(data_dir+"prepared/word_idx_english.json") as f:
        mapping = json.load(f)
        rev_mapping = {v: k for k, v in mapping.items()}

    return [
        [rev_mapping[t] for t in sent if t != 0] for sent in test_X
    ]


def evaluate(runs, data_dir, model_dir, domain, command, template):
    ae_data = np.load(data_dir+"prepared/dataset_english_annotated.npz")
    raw_X = recreate_data(data_dir, ae_data['test_X'])
    results = []
    for r in range(runs):
        model = torch.load(model_dir+"xu_"+str(r))
        result = test(model, ae_data['test_X'], raw_X, domain, command, template, crf=False)
        results.append(result)
    print(sum(results)/len(results))


parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--data_dir', type=str, default="../data/english/")
parser.add_argument('--model_dir', type=str, default="../data/english/models/")
parser.add_argument('--domain', type=str, default="restaurant")

args = parser.parse_args()

if args.domain == 'restaurant':
    command = "java --add-modules java.xml.bind -cp script/A.jar absa16.Do Eval -prd data/official_data/pred.xml -gld data/official_data/EN_REST_SB1_TEST.xml.gold -evs 2 -phs A -sbt SB1"
    template = "data/official_data/EN_REST_SB1_TEST.xml.A"
elif args.domain == 'laptop':
    command = "java -cp script/eval.jar Main.Aspects data/official_data/pred.xml data/official_data/Laptops_Test_Gold.xml"
    template = "data/official_data/Laptops_Test_Data_PhaseA.xml"

evaluate(args.runs, args.data_dir, args.model_dir, args.domain, command, template)
