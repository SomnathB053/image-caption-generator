from torchtext.data import get_tokenizer
from .dataset import image_caption_dict
import torchvision.models as models
from .model import ImageCaptionGen
import torch
import torch.nn as nn
import torchvision.transforms as T

def infer(image):
    tokenizer = get_tokenizer('basic_english')
    dataframe, vocab = image_caption_dict('dataset/annot/Flickr8k.token.txt', tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet =  models.resnet50(weights= models.ResNet50_Weights.IMAGENET1K_V2)
    resnet_backbone = torch.nn.Sequential(*(list(resnet.children())[:-1])+ [nn.Flatten()])
    model = ImageCaptionGen(400,resnet_backbone, len(vocab),32, 512, 2, 8, 1024, 0.1).to(device)
    try:
        model.load_state_dict(torch.load('imCgen.pt'))
    except:
        print("ImCgen.pt not found. infering from untrained model")
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
        caption = torch.LongTensor(decoder_input).unsqueeze(0).to(device)
        mask = torch.zeros((1, max_len), dtype=torch.bool).to(device)
        mask[:,i+1:] = True
        with torch.no_grad():
            pred = model(image, caption, mask)
        idx = torch.argmax(pred, dim=-1, keepdim=False)[:,i].item()
        if vocab.get_itos()[idx]== '<eos>':
            break
        output.append(vocab.get_itos()[idx])

    print(output)




        



