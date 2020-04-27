from transformers import BertForTokenClassification, BertJapaneseTokenizer, get_linear_schedule_with_warmup
from flask import Flask, render_template, request
import argparse
import json
from predict import predict
import torch
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from utils import iob2json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__)

@app.route("/", methods=['POST'])
def index():
    word = request.form["text"]
    input_json = json.loads(word)
    input_x = [input_json[str(i)] for i in range(len(input_json))]
    input_x = [tokenizer.tokenize(t) for t in input_x]
    input_x = [tokenizer.convert_tokens_to_ids(['[CLS]'] + x) for x in input_x]
    tags = predict(model, input_x,  device)

    labels = [[id2label[t] for t in tag] for tag in tags]
    input_x = [tokenizer.convert_ids_to_tokens(t)[1:] for t in input_x]

    output = [zip(x, l) for x, l in zip(input_x, labels)]
    output = [iob2json.decode_iob(i) for i in output]

    return iob2json.create_json(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BERT')
    parser.add_argument('--model_dir', type=str, help='data path')
    args = parser.parse_args()

    tokenizer = BertJapaneseTokenizer.from_pretrained("bert-base-japanese-char")

    with open(args.model_dir + '/label_vocab.json', 'r') as f:
        label_vocab = json.load(f)
    id2label = {v:k for k, v in label_vocab.items()}

    model = BertForTokenClassification.from_pretrained('bert-base-japanese-char', num_labels=len(label_vocab))

    model_path = args.model_dir + '/final.model'
    model.load_state_dict(torch.load(model_path))

    app.run(port='8000', host='0.0.0.0', debug=True)
