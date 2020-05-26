import torch
import mojimoji

def load_dataset(path):
    with open(path, 'r') as f:
        lines = f.read()

    lines = lines.split("\n\n")
    lines = [[token for token in l.split("\n") if token != ""] for l in lines if l != ""]

    data = []
    label = []
    for line in lines:
        sent = []
        sent_label = []
        for l in line:
            token, tag = l.split("\t")
            sent.append(mojimoji.zen_to_han(token, kana=False))
            sent_label.append(tag)

        data.append(sent)
        label.append(sent_label)

    return data, label

def create_vocab(data):
    vocab = {}
    vocab["[PAD]"] = len(vocab)
    vocab["[UNK]"] = len(vocab)

    for d in data:
        for token in d:
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab

def create_label_vocab(label):
    vocab = {}
    vocab["[PAD]"] = len(vocab)

    for l in label:
        for token in l:
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab

def sent2input(sent, vocab):
    return [vocab[token] if token in vocab else vocab["[PAD]"] for token in sent]

def data2input(data, vocab):
    return [sent2input(sent, vocab) for sent in data]

def pad_sentence(sent, length, pad_value=0):
    return sent + [pad_value] * (length - len(sent)) if len(sent) <= length else sent[:length]

def pad_sequence(seq, max_length=512, pad_value=0, issort=True):
    length = len(seq[0]) if issort else len(sorted(seq, key=lambda x: len(x), reverse=True)[0])
    max_length = min(length, max_length)
    return [pad_sentence(s, max_length, pad_value) for s in seq]

class Batch(object):
    def __init__(self, data, label, batch_size=8, pad_value=0, max_size=512, device=None, sort=True):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.pad_value = pad_value
        self.max_size = max_size
        self.device = device
        self.sort = sort
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        data = zip(self.data, self.label)
        if self.sort:
            data = sorted(data, key=lambda x: len(x[0]), reverse=True)
        else:
            data = list(data)

        for i in range(0, len(data), self.batch_size):
            s_pos = i
            e_pos = min(i+self.batch_size, len(data))

            x = [d[0] for d in data[s_pos:e_pos]]
            l = [d[1] for d in data[s_pos:e_pos]]
            length = [len(d[0]) for d in data[s_pos:e_pos]]
            x = pad_sequence(x, self.max_size, pad_value=self.pad_value, issort=self.sort)
            l = pad_sequence(l, self.max_size, pad_value=self.pad_value, issort=self.sort)

            yield x, l, length

class Mydataset(object):
    def __init__(self, path, vocab=None, label_vocab=None, batch_size=8):
        self.data, self.label = load_dataset(path)
        self.vocab = vocab if vocab is not None else create_vocab(self.data)
        self.label_vocab = label_vocab if label_vocab is not None else create_label_vocab(self.label)
        self.x, self.l = data2input(self.data, self.vocab), data2input(self.label, self.label_vocab)
        self.batch_size = batch_size

    def get_vocab(self):
        return self.vocab

    def get_label_vocab(self):
        return self.label_vocab

    def __iter__(self):
        data = zip(self.x, self.l)
        data = sorted(data, key=lambda x: len(x[0]), reverse=True)

        for i in range(0, len(data), self.batch_size):
            s_pos = i
            e_pos = min(i+self.batch_size, len(data))

            x = [d[0] for d in data[s_pos:e_pos]]
            l = [d[1] for d in data[s_pos:e_pos]]
            length = [len(d[0]) for d in data[s_pos:e_pos]]
            x = pad_sequence(x, pad_value=self.vocab["[PAD]"])
            l = pad_sequence(l, pad_value=self.label_vocab["[PAD]"])

            yield x, l, length



if __name__ == "__main__":
    """
    data, label = load_dataset("data/sample_iob.iob")
    vocab = create_vocab(data)
    label_vocab = create_vocab(label)
    input = data2input(data, vocab)
    label_input = data2input(label, label_vocab)
    batch = Batch(input, label_input, 8, 0)
    """

    dataset = Mydataset("../data/sample_iob.iob")
    for x, l, leng in dataset:
        print(x)
        print(l)
        print(leng)
