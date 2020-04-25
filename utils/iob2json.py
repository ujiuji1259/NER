import argparse
import json

def read_iob(fn):
    with open(fn, 'r') as f:
        lines = [line.split('\n') for line in f.read().split('\n\n') if line != '']
        iobs = [[i.split('\t')[:2] for i in line if i != ''] for line in lines]
    return iobs

def decode_iob(sent):
    sentence = []
    tags = []

    current_tag = ''
    s_pos = 0
    e_pos = 0
    for idx, iob in enumerate(sent):
        sentence.append(iob[0])
        if iob[1][0] == 'B':
            tag = iob[1].split('-')[-1]
            if not current_tag:
                s_pos = idx
                current_tag = tag
            elif current_tag != tag:
                e_pos = idx
                tags.append((current_tag, s_pos, e_pos))
                s_pos = idx
                current_tag = tag
        elif iob[1][0] == 'O':
            if current_tag:
                e_pos = idx
                tags.append((current_tag, s_pos, e_pos))
                current_tag = ''

    return ''.join(sentence), tags

def create_json(docs):
    output = {str(i):{'sent':sent[0], 'tags':list(sent[1])} for i,sent in enumerate(docs)}
    return json.dumps(output, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert text to IOB2 format.')
    parser.add_argument('--input', default=None, help='input file path　(Mandatory)')
    parser.add_argument('--output', default=None, help='output file path')

    args = parser.parse_args()
    iobs = read_iob(args.input)

    docs = [decode_iob(i) for i in iobs]
    with open(args.output, 'w') as f:
        f.write(create_json(docs))

