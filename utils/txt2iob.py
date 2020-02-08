from xml.etree.ElementTree import iterparse
import xml.etree.ElementTree as ET
import MeCab
import argparse

def sent2iob(sent, format="c", tag_list=None):
    text = '<sent>' + sent + '</sent>'
    parser = ET.XMLPullParser(['start', 'end'])
    parser.feed(text)

    ne_type = "O"
    ne_prefix = ""
    res = ""
    label = []
    tag_set = set()
    for event, elem in parser.read_events():
        if event == "start":
            assert len(tag_set) < 2, "タグが入れ子になっています\n{}".format(sent)
            word = elem.text if elem.text is not None else ""
            res += word

            isuse = tag_list is None or (tag_list is not None and elem.tag in tag_list)
            if elem.tag != "sent" and isuse:
                tag_set.add(elem.tag)
                label += [elem.tag] * len(word)
            else:
                label += ["O"] * len(word)

        if event == "end":
            if elem.tag != "sent" and isuse:
                tag_set.remove(elem.tag)
            word = elem.tail if elem.tail is not None else ""
            res += word
            label += ["O"] * len(word)

    if format == "c":
        res = list(res)
    elif format == "w":
        mecab = MeCab.Tagger('-Owakati')
        res = mecab.parse(res)[:-1].split(' ')[:-1]

    cnt = 0
    output = []
    prev = "O"
    post = ""
    for token in res:
        assert len(set(label[cnt:cnt+len(token)])) == 1, "形態素とラベルが食い違っています\n{2}\n{0} : {1}".format(token, label[cnt:cnt+len(token)], "".join(res))
        pre_token = ""

        if label[cnt] != "O" and (prev == "O" or prev != label[cnt]):
            pre_token = "B-"
        elif label[cnt] != "O" and prev == label[cnt]:
            pre_token = "I-"

        prev = label[cnt]

        output.append(token + "\t" + pre_token + label[cnt])
        cnt += len(token)

    return "\n".join(output)

def doc2iob(doc, format="c", tag_list=None):
    output = [sent2iob(s.replace("\n", ""), format, tag_list) for s in doc]
    return "\n\n".join(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert text to IOB2 format.')
    parser.add_argument('--format', default="c", help='Unit : c (char) or w (word)')
    parser.add_argument('--tag', default=None, help='target tag : C,M')
    parser.add_argument('--input', default=None, help='input file path　(Mandatory)')
    parser.add_argument('--output', default=None, help='output file path')

    args = parser.parse_args()

    tag = args.tag.split(",") if args.tag is not None else None

    assert args.input is not None, "入力ファイルが指定されていません"

    with open(args.input, "r") as f:
        doc = f.readlines()

    output = doc2iob(doc, format=args.format, tag_list=tag)

    path = ".".join(args.input.split(".")[:-1]) + "_iob.iob" if args.output is None else args.output

    with open(path, "w") as f:
        print("output : ", path)
        f.write(output)
