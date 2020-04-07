import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from crf import CRF
from dataset import tokenizer, VOCAB, tag2idx, idx2tag
from sklearn.metrics import classification_report

def train(model, iterator, optimizer, criterion, epoch):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y 
        optimizer.zero_grad()
        logits_focal, logits_crf, y, _ = model(x, y, epoch)

        logits_focal = logits_focal.view(-1, logits_focal.shape[-1])
        y_focal = y.view(-1)

        loss1 = criterion(logits_focal, y_focal)
        loss = logits_crf+loss1
        loss.backward()

        optimizer.step()

        if i==0:
            print("==============Check Dataloader===============")
            print("words:", words[0])
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            print("y:", _y.cpu().numpy()[0][:seqlens[0]])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])
            print("=============================================")
        

def eval(model, iterator, epoch=None):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat, Y_hat_viterbi = [], [], [], [], [], []

    crf = CRF(len(VOCAB))
    crf.to(device='cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            logits_focal, _, _, y_hat = model(x, y)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            

    all_preds = []
    with open("results.txt", 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            all_preds.append(preds)
            assert len(preds)==len(words.split())==len(tags.split()), "Sentence: {}\n True Tags: {}\n Pred Tags: {}".format(words,tags,preds)
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write("{} {} {}\n".format(w,t,p))
            fout.write("\n")

    y_true =  np.array([tag2idx[line.split()[1]] for line in open("results.txt", 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split()[2]] for line in open("results.txt", 'r').read().splitlines() if len(line) > 0])

    if epoch is not None:
        report = classification_report(y_true,y_pred, labels = [1,2,3,4,5,6,7,8,9], digits=4, zero_division=0)
        print(f"============Evaluation at Epoch={epoch}============")
        print(report)
    os.remove("results.txt")

    return all_preds

