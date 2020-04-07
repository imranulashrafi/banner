from net import Net
from dataset import NerDataset, pad, VOCAB
from trainer import eval
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='NER Inference')
parser.add_argument("--sent", nargs=1, required=True, type=str, help="Enter bangla sentence")

args = parser.parse_args()

def run_ner_infer(sent):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    top_rnns=True
    model = Net(top_rnns, len(VOCAB), device, finetuning=True)
    if device == 'cpu':
    	model.load_state_dict(torch.load('models/banner_model.pt', map_location=torch.device('cpu')))
    elif device == 'cuda':
    	model.load_state_dict(torch.load('.models/banner_model.pt'))
    model.to(device)

    tags = []
    for x in range(len(sent.split())):
        tags.append('O')
    sent_infer=[]
    sent_infer.append(["[CLS]"] + sent.split() + ["[SEP]"])
    tags_infer=[]
    tags_infer.append(["<PAD>"] + tags + ["<PAD>"])

    infer_data = NerDataset(sent_infer, tags_infer)

    infer_iter = torch.utils.data.DataLoader(dataset=infer_data,
                             batch_size=1,
                             shuffle=False,
                             collate_fn = pad,
                             num_workers=0
                             )
    pred = eval(model, infer_iter)
    for x in range(len(pred[0])):
        if pred[0][x] == '<PAD>':
            pred[0][x] = 'O'
    return sent_infer[0][1:-1],pred[0][1:-1]


def main():
    sent = ''.join(args.sent)
    print(run_ner_infer(sent))


if __name__ == '__main__':
    main()

