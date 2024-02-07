'''
create dataset
split dataset
create dataloader
epoch iter
train 
'''
from torchtext.data import get_tokenizer
from dataset import image_caption_dict, Image_caption_dataset
from torch.utils.data import DataLoader
import torchvision.models as models
from model import ImageCaptionGen
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

TRAIN_TEST_SPLIT = 0.8


resnet =  models.resnet50(weights= models.ResNet50_Weights.IMAGENET1K_V2)
resnet_backbone = torch.nn.Sequential(*(list(resnet.children())[:-1])+ [nn.Flatten()])



'''TEST_DATAFRAME = dataframe[int(len(dataframe)*TRAIN_TEST_SPLIT):]
TEST_DATASET = Image_caption_dataset(tokenizer, vocab, TEST_DATAFRAME, 32)
TEST_DATALOADER =  DataLoader(TEST_DATASET, 16, shuffle=True)
'''


def train(train_split=TRAIN_TEST_SPLIT, EPOCHS=1):
    tokenizer = get_tokenizer('basic_english')
    dataframe, vocab = image_caption_dict('dataset/annot/Flickr8k.token.txt', tokenizer)
    TRAIN_DATAFRAME = dataframe[:int(len(dataframe)*train_split)]
    TRAIN_DATASET = Image_caption_dataset(tokenizer, vocab, TRAIN_DATAFRAME, 32)
    TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, 32, shuffle=True)

    model = ImageCaptionGen(400,resnet_backbone, len(vocab),32, 512, 1, 2, 512, 0)
    criterion =nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(EPOCHS):
        print("beginning epoch 1...")
        for _, data in enumerate(tqdm(TRAIN_DATALOADER)):
            images = data['image']
            print('Image tensor shape: ', images.shape)
            tokens = data['captions_tokens']
            print('Token tensor shape: ', tokens.shape)
            masks = data['pad_ignore_mask']
            print('Mask tensor shape', masks.shape)

            tokens_sub_batches = torch.split(tokens, 1, 1)
            masks_sub_batches = torch.split(masks, 1, 1)
            
            avg_loss=0
            for sub_batch in range(len(tokens_sub_batches)):
                print(f'training sub batch {epoch}- {_}-{sub_batch}...')
                pred = model(images, tokens_sub_batches[sub_batch].squeeze(1)[:, :-1], masks_sub_batches[sub_batch].squeeze(1)[:, :-1])
                print("pred shape: ", pred.shape, type(pred))
                pred = pred.contiguous().view(-1, len(vocab))

                print("reshaped pred shape: ", pred.shape)
                labels = tokens_sub_batches[sub_batch].squeeze(1)[:, 1:]
                print('labels shape', labels.shape, type(labels))
                labels = labels.contiguous().view(-1)
                
                loss = criterion(pred, labels)
                avg_loss+= loss.item()
                loss.backward()
                optimizer.step()
            print('print avg loss', avg_loss/5)
            avg_loss=0


train()

                


