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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


resnet =  models.resnet50(weights= models.ResNet50_Weights.IMAGENET1K_V2)
resnet_backbone = torch.nn.Sequential(*(list(resnet.children())[:-1])+ [nn.Flatten()])


def train(train_split=TRAIN_TEST_SPLIT, EPOCHS=1):
    tokenizer = get_tokenizer('basic_english')
    dataframe, vocab = image_caption_dict('dataset/annot/Flickr8k.token.txt', tokenizer)
    TRAIN_DATAFRAME = dataframe[:int(len(dataframe)*train_split)]
    TRAIN_DATASET = Image_caption_dataset(tokenizer, vocab, TRAIN_DATAFRAME, 32)
    TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, 64, shuffle=False)

    model = ImageCaptionGen(img_size=400,cnn_backbone=resnet_backbone, vocab_len=len(vocab), seq_len=32, d_model=512, n_decode=2, n_head=8, fc_dim=1024,dropout= 0.1)
    model.to(device)

    model.train()
    print(f'-------------------COMPUTE DEVICE IS {device}------------------------')
    criterion =nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000005)

    for epoch in range(EPOCHS):
        print("beginning epoch 1...")
        batch_loader = tqdm(enumerate(TRAIN_DATALOADER), desc= 'begining batch training', postfix='current_loss = NA ')
        for _, data in batch_loader:
            images = data['image'].to(device)
            tokens = data['captions_tokens']
            masks = data['pad_ignore_mask']
            tokens_sub_batches = torch.split(tokens, 1, 1)
            masks_sub_batches = torch.split(masks, 1, 1)
            avg_loss=0
            for sub_batch in range(len(tokens_sub_batches)):
                batch_loader.set_description_str(f'Training sub-batch {epoch}--{_}--{sub_batch}->->->', refresh=True)
                tokens_sub_batches_i = tokens_sub_batches[sub_batch].squeeze(1)[:, :-1].to(device)
                masks_sub_batches_i = masks_sub_batches[sub_batch].squeeze(1)[:, :-1].to(device)
                pred = model(images, tokens_sub_batches_i, masks_sub_batches_i)
                pred = pred.contiguous().view(-1, len(vocab))
                labels = tokens_sub_batches[sub_batch].squeeze(1)[:, 1:]
                labels = labels.contiguous().view(-1).to(device)
                loss = criterion(pred, labels)
                batch_loader.set_postfix_str(f'Current loss: {loss.item()}')
                avg_loss+= loss.item()
                loss.backward()
                optimizer.step()
            avg_loss=0
    torch.save(model.state_dict(),'imCgen.pt' )


if __name__ == '__main__':
    train()
    print("Training successful. saved weights as imCgen.pt")

                


