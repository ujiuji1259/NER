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

from utils import data_utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, x, y, max_epoch=10, lr=3e-5, batch_size=8, val=None, outputdir=None):
    data = data_utils.Batch(x, y, batch_size=batch_size)
    if val is not None:
        val_data = data_utils.Batch(val[0], val[1], batch_size=batch_size)
        val_loss = []

    loss = nn.NLLLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_step = int((len(data)//batch_size)*max_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_step*0.1), total_step)

    losses = []
    min_val_loss = 999999999999
    model.to(device)
    for epoch in tqdm(range(max_epoch)):
        print('EPOCH :', epoch+1)
        model.train()
        all_loss = 0
        step = 0

        for sent, label, _ in data:
            sent = torch.tensor(sent).to(device)
            label = torch.tensor(label).to(device)
            mask = [[float(i>0) for i in ii] for ii in sent]
            mask = torch.tensor(mask).to(device)

            output = model(sent, attention_mask=mask, labels=label)
            loss = output[0]
            all_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            step += 1

        losses.append(all_loss / step)
        print(losses)

        if val is not None:
            model.eval()
            all_loss = 0
            step = 0

            for sent, label, _ in val_data:
                sent = torch.tensor(sent).to(device)
                label = torch.tensor(label).to(device)
                mask = [[float(i>0) for i in ii] for ii in sent]
                mask = torch.tensor(mask).to(device)

                output = model(sent, attention_mask=mask, labels=label)
                loss = output[0]
                all_loss += loss.item()

                step += 1
            val_loss.append(all_loss / step)
            output_path = outputdir + '/checkpoint{}.model'.format(len(val_loss)-1)
            torch.save(model.state_dict(), output_path)

    if val is not None:
        min_epoch = np.argmin(val_loss)
        print(min_epoch)
        model_path = outputdir + '/checkpoint{}.model'.format(min_epoch)
        model.load_state_dict(torch.load(model_path))

    torch.save(model.state_dict(), outputdir+'/final.model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BERT')
    parser.add_argument('--train_path', type=str, help='data path')
    parser.add_argument('--val_path', type=str, help='data path')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--output_dir', type=str, help='batch size')
    args = parser.parse_args()

    tokenizer = BertJapaneseTokenizer.from_pretrained("bert-base-japanese-char")

    train_data = data_utils.load_dataset(args.train_path)

    label_vocab = data_utils.create_label_vocab(train_data[1])
    with open(args.output_dir + '/label_vocab.json', 'w') as f:
        json.dump(label_vocab, f, ensure_ascii=False)

    input_x = [tokenizer.convert_tokens_to_ids(['[CLS]'] + x) for x in train_data[0]]
    input_y = [data_utils.sent2input(['[PAD]']+x, label_vocab) for x in train_data[1]]

    val_data = data_utils.load_dataset(args.train_path)
    input_x_val = [tokenizer.convert_tokens_to_ids(['[CLS]'] + x) for x in val_data[0]]
    input_y_val = [data_utils.sent2input(['[PAD]']+x, label_vocab) for x in val_data[1]]

    model = BertForTokenClassification.from_pretrained('bert-base-japanese-char', num_labels=len(label_vocab))

    train(model, input_x, input_y, val=[input_x_val, input_y_val], outputdir=args.output_dir)

