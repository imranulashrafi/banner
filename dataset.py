import numpy as np
import torch
from torch.utils import data
from pytorch_pretrained_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

VOCAB = ('<PAD>', 'I-LOC', 'B-ORG', 'O', 'I-OBJ', 'I-PER', 'B-OBJ', 'I-ORG', 'B-LOC', 'B-PER')

tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

class NerDataset(data.Dataset):
    def __init__(self, sents, tags_li):
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] 


        x, y = [], [] 
        is_heads = [] 
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<PAD>"] * (len(tokens) - 1) 
            yy = [tag2idx[each] for each in t] 

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        seqlen = len(y)

        
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


def pad(batch):
    
    f = lambda x: [sample[x] for sample in batch]
    #x = f(1)
    #y = f(-2)
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] 
    x = f(1, maxlen)
    y = f(-2, maxlen)


    f = torch.LongTensor
    
    return words, f(x), is_heads, tags, f(y), seqlens