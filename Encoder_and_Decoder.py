import torch
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN_Encoder(torch.nn.Module):
    def __init__(self, encoded_image_size=14):
        super(CNN_Encoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete the last FC and pool layer
        self.resnet = torch.nn.Sequential(*modules)  # * is used to split the generator to single element
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        # images: batch_size x 3 x width x height (width,height is 256)
        # out : batch_size x 2048 x width/3 x height/3 (out is 8)
        with torch.no_grad():
            out = self.resnet(images)
            out = self.adaptive_pool(out)  # transform the out's width and height to 14
            out = out.permute(0, 2, 3, 1)  # batch_size x encoded_size x encoded_size x channel(2048)
        return out


class Attention(torch.nn.Module):  # Attention mechanism to calculate the weights
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_linear_transform = torch.nn.Linear(encoder_dim, attention_dim)  # Flatten the encoder_out
        self.decoder_linear_transform = torch.nn.Linear(decoder_dim, attention_dim)
        self.attention_value = torch.nn.Linear(attention_dim, 1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att_1 = self.encoder_linear_transform(encoder_out)  # batch_size x num_pixels(14x14) x attention_dim
        att_2 = self.decoder_linear_transform(decoder_hidden)  # batch_size x attention_dim
        att_2 = att_2.unsqueeze(1)
        att = torch.relu(att_1 + att_2)
        att = self.attention_value(att)
        att = att.squeeze(2)  # batch_size x num_pixels
        alpha = self.nn.Softmax(att)  # calculate the weights for pixels
        encoding_with_attention = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # batch_size x encoder_dim
        return encoding_with_attention,alpha

class Decoder_with_attention(torch.nn.Module):
    def __init__(self,attention_dim,embed_dim,decoder_dim,vocab_size,encoder_dim=2048,dropout=0.5):
        super(Decoder_with_attention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim

        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)

        self.embedding = torch.nn.Embedding(vocab_size,embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout)