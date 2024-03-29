from collections import Counter
import torchtext
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
import torch
import cv2
import os

def image_caption_dict(FILENAME, tokenizer):
    caption_map= {}
    corpus = []
    with open(FILENAME) as f:
        for line in f:
            line = line.rstrip('\n')
            line = line.rstrip(' .')
            img_name, caption = line.split('\t')
            
            img_name = img_name.split('.')[0] + '.jpg'

            if img_name == "2258277193_586949ec62.jpg":
                continue

            if img_name in caption_map.keys():
                caption_map[img_name] =caption_map[img_name] + [caption]
            else:
                caption_map[img_name] = [caption]
            corpus.append(caption)
    counter = Counter()
    for item in corpus:
        counter.update(tokenizer(item))
    vocab = torchtext.vocab.vocab(counter)
    for sp_token, index in zip(['<unk>', '<pad>', '<sos>', '<eos>'], [0,1,2,3]):
        vocab.insert_token(sp_token, index)
        vocab.set_default_index(0)
    dataframe = pd.DataFrame()
    dataframe = pd.DataFrame(columns=["image_filename", "captions"])
    dataframe["image_filename"] = caption_map.keys()
    dataframe["captions"] = dataframe["image_filename"].map(
        lambda x: caption_map[x]
    )
    return dataframe, vocab
class Image_caption_dataset(Dataset):
    def __init__(self, tokenizer, vocab, dataframe, seq_len):
        super().__init__()
        self.tokenizer = tokenizer
        self. dataframe = dataframe
        self.seq_len = seq_len+1
        self.vocab = vocab
        self.transform = T.Compose([
            T.ToTensor(),
            T.Pad((150)),
            T.CenterCrop((400,400)),
        ])
        self.image_folder = os.path.join(os.getcwd(), 'dataset', 'images', 'Flicker8k_Dataset')

    def __getitem__(self, index):
        pad_idx = self.vocab["<pad>"]
        sos_idx = self.vocab["<sos>"]
        eos_idx = self.vocab["<eos>"]
        pad_starts = None
        if torch.is_tensor(index):
            index = index.item()
        image_name, caption_list = self.dataframe.iloc[index]
        image = cv2.imread(os.path.join(self.image_folder,image_name))
        tokens_array = []
        pad_ignore_mask= []
        for idx, e in enumerate(caption_list):
            tokens  = self.vocab(self.tokenizer(e))
            tokens = [sos_idx] + tokens +[eos_idx]

            if len(tokens) < self.seq_len:
                pad_starts = len(tokens)
                tokens = tokens + [pad_idx]*(self.seq_len - len(tokens))
            else:
                tokens = tokens[:self.seq_len-1] + [eos_idx]
            tokens_array.append(tokens)
            mask = torch.zeros(self.seq_len)
            if pad_starts is not None:
                mask[pad_starts:] = True
            pad_ignore_mask.append(mask)
        tokens_array = torch.LongTensor(tokens_array)
        pad_ignore_mask = torch.stack(pad_ignore_mask)

        assert image is not None, f'image empty {image_name}'
        assert tokens_array is not None, 'token empty'
        assert caption_list is not None, 'list empty'
        assert pad_ignore_mask is not None, 'mask empty'
        out_dict = {
            "image": self.transform(image),
            "captions_tokens": tokens_array,
            "captions": caption_list,
            "pad_ignore_mask": pad_ignore_mask,
        }
        return out_dict
    def __len__(self):
        return len(self.dataframe)       
