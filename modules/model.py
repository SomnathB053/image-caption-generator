import torchvision.models as models
import torch
import torch.nn as nn




class ImageCaptionGen(nn.Module):
    def __init__(self, img_size, cnn_backbone, vocab_len,  seq_len, d_model, n_decode, n_head, fc_dim, dropout):
        super().__init__()
        self.vocab_len= vocab_len
        self.seq_len = seq_len
        self.d_model = d_model
        self.img_size = img_size

        self.encoder = cnn_backbone 
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        self.embedding = nn.Embedding(self.vocab_len, self.d_model)
        dummy_img = torch.rand([3,img_size,img_size], requires_grad=False).unsqueeze(0) # shape [batch, channel, img_size, img_size]

        self.encoder_linear = nn.Linear(cnn_backbone(dummy_img).shape[1], self.d_model)  # [batch, cnn_out] -> [batch, d_model]

        self.embeddidng = nn.Embedding(vocab_len, d_model) # [batch, seq_len, 1] -> [batch, seq_len, d_model]
        self.pos_embedding = nn.Embedding(seq_len, d_model) # [batch, seq_len, 1] -> [batch, seq_len, d_model]

        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, fc_dim, dropout, batch_first= True)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_decode)

        self.linear = nn.Linear(d_model, vocab_len)


    def forward(self, image, caption, ignore_pad_mask=None):
        #batch_size, seq_len = caption.shape[0], caption.shape[1]

        scale = torch.sqrt(torch.tensor([self.d_model]))

        x = self.embeddidng(caption) * scale
        pos = self.pos_embedding(torch.zeros_like(caption).long())
        x= x + pos

        with torch.no_grad():
            image_rep = self.encoder(image)
        assert image_rep.dim() == 2, 'dimetions of image output not 2'
        image_rep = self.encoder_linear(image_rep) # [batch, encoder_out_dim] -> [batch_d_model]

        decoder_input = image_rep.unsqueeze(1) # [batch, d_model] -> [batch, 1, d_model] need for calculating the attention

        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])


        x = self.decoder(x, decoder_input, tgt_mask = causal_mask, tgt_key_padding_mask = ignore_pad_mask) #[batch, seq_len, d_model]
        

        out = self.linear(x)
        return out

