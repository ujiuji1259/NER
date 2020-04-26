import sys
import argparse
from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from utils import txt2iob
from transformers import BertJapaneseTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BERT')
    parser.add_argument('--path', type=str, help='data path')
    parser.add_argument('--output_path', type=str, help='data path')
    parser.add_argument('--tag', default=None, help='valid tag list : C,M')
    args = parser.parse_args()
    tag = args.tag.split(",") if args.tag is not None else None

    tokenizer = BertJapaneseTokenizer.from_pretrained("bert-base-japanese-char")

    with open(args.path, 'r') as f:
        lines = [line for line in f.read().split('\n') if line != '']

    output = '\n\n'.join(['\n'.join(['\t'.join(t) for t in line]) for line in txt2iob.doc2iob(lines, format=tokenizer.tokenize, tag_list=tag)])
    with open(args.output_path, 'w') as f:
        f.write(output)

