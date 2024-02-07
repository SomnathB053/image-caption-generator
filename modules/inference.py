from torchtext.data import get_tokenizer
from .dataset import image_caption_dict, Image_caption_dataset
from torch.utils.data import DataLoader
import torchvision.models as models
from .model import ImageCaptionGen
import torch
import torch.nn as nn
import torchvision.transforms as T






def infer(image):
    tokenizer = get_tokenizer('basic_english')
    dataframe, vocab = image_caption_dict('dataset/annot/Flickr8k.token.txt', tokenizer)

    resnet =  models.resnet50(weights= models.ResNet50_Weights.IMAGENET1K_V2)
    resnet_backbone = torch.nn.Sequential(*(list(resnet.children())[:-1])+ [nn.Flatten()])
    model = ImageCaptionGen(400,resnet_backbone, len(vocab),32, 512, 1, 2, 512, 0.1)
    model.eval()
    transf =T.Compose([
            T.ToTensor(),
            T.Pad((150)),
            T.CenterCrop((400,400))
    ])

    image = transf(image)
    image = image.unsqueeze(0)
    assert image.dim() == 4, "image dimetions should be 4"
    max_len =32
    decoder_input = vocab(['<sos>']) + vocab(['<pad'])*(max_len-1)

    output = []

    for i in range(max_len):
        caption = torch.LongTensor(decoder_input).unsqueeze(0)
        mask = torch.zeros((1, max_len), dtype=torch.bool)

        mask[:,i+1:] = True
        print(caption.shape, mask.shape)
        with torch.no_grad():
            pred = model(image, caption, mask)
        
        idx = torch.argmax(pred, dim=-1, keepdim=False)[:,i].item()
        #print(idx.shape)

        if vocab.get_itos()[idx]== '<eos>':
            break
        output.append(vocab.get_itos()[idx])


    print(output)




        



