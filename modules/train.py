'''
create dataset
split dataset
create dataloader
epoch iter
train 
'''
from torchtext.data import get_tokenizer
from modules.dataset import image_caption_dict, Image_caption_dataset
from torch.utils.data import DataLoader


TRAIN_TEST_SPLIT = 0.8


tokenizer = get_tokenizer('basic_english')

dataframe, vocab = image_caption_dict('dataset/annot/Flickr8k.token.txt', tokenizer)


TRAIN_DATAFRAME = dataframe[:int(len(dataframe)*TRAIN_TEST_SPLIT)]
TEST_DATAFRAME = dataframe[int(len(dataframe)*TRAIN_TEST_SPLIT):]

TRAIN_DATASET = Image_caption_dataset(tokenizer, vocab, TRAIN_DATAFRAME, 32)
TEST_DATASET = Image_caption_dataset(tokenizer, vocab, TEST_DATAFRAME, 32)

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, 16, shuffle=True)
TEST_DATALOADER =  DataLoader(TEST_DATASET, 16, shuffle=True)
