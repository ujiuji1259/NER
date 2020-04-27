from transformers import BertForTokenClassification, BertJapaneseTokenizer, get_linear_schedule_with_warmup
import argparse
from torchtext import data, datasets
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from utils import data_utils, iob2json
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(model, x, outputpath=None, device=None):
    data = data_utils.Batch(input_x, input_x, batch_size=8)

    model.to(device)
    model.eval()

    res = []

    for sent, _, _ in data:
        sent = torch.tensor(sent).to(device)
        mask = [[float(i>0) for i in ii] for ii in sent]
        mask = torch.tensor(mask).to(device)

        output = model(sent, attention_mask=mask)
        logits = output[0].detach().cpu().numpy()
        tags = np.argmax(logits, axis=2)[:, 1:].tolist()
        res.extend(tags)

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BERT')
    parser.add_argument('--model_dir', type=str, help='data path')
    parser.add_argument('--output_path', type=str, help='batch size')
    parser.add_argument('--input_path', type=str, help='batch size')
    args = parser.parse_args()

    tokenizer = BertJapaneseTokenizer.from_pretrained("bert-base-japanese-char")


    with open(args.input_path, 'r') as f:
        train_data = [line for line in f.read().split('\n') if line != '']
    train_data = [tokenizer.tokenize(t) for t in train_data]
    print(train_data)

    with open(args.model_dir + '/label_vocab.json', 'r') as f:
        label_vocab = json.load(f)
    id2label = {v:k for k, v in label_vocab.items()}

    input_x = [tokenizer.convert_tokens_to_ids(['[CLS]'] + x) for x in train_data]

    model = BertForTokenClassification.from_pretrained('bert-base-japanese-char', num_labels=len(label_vocab))

    model_path = args.model_dir + '/final.model'
    model.load_state_dict(torch.load(model_path))

    tags = predict(model, input_x, args.output_path, device)
    labels = [[id2label[t] for t in tag] for tag in tags]
    input_x = [tokenizer.convert_ids_to_tokens(t)[1:] for t in input_x]

    output = []
    for x, t in zip(input_x, labels):
        output.append('\n'.join([x1 + '\t' + str(x2) for x1, x2 in zip(x, t)]))

    with open(args.output_path, 'w') as f:
        f.write('\n\n'.join(output))

