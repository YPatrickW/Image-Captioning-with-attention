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
        out = self.resnet(images)
        out = self.adaptive_pool(out)  # transform the out's width and height to 14
        out = out.permute(0, 2, 3, 1)  # batch_size x encoded_size x encoded_size x channel(2048)
        return out

    def Calculate_grads(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for Conv_block in list(self.resnet.children())[5:]:
            for p in Conv_block.parameters():
                p.requires_grad = fine_tune


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
        alpha = self.nn.Softmax(att)  # calculate the weights for pixels (batch_size x num_pixels)
        encoding_with_attention = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # batch_size x encoder_dim
        return encoding_with_attention, alpha


class Decoder_with_attention(torch.nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(Decoder_with_attention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.dropout = dropout
        self.dropout = torch.nn.Dropout(p=self.dropout)

        self.attention = Attention(encoder_dim, decoder_dim,
                                   attention_dim)  # Attention net to calculate the weight for pixels

        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.decode_step = torch.nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = torch.nn.Linear(encoder_dim, decoder_dim)
        self.init_c = torch.nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = torch.nn.Linear(decoder_dim, encoder_dim)
        self.fc = torch.nn.Linear(decoder_dim, vocab_size)
        self.sigmoid = torch.nn.Sigmoid()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # batch_size x decoder_dim
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions, caption_lengths):
        batch_size = encoder_out.shape[0]
        encoder_dim = encoder_out.shape[3]  # 2048
        vocab_size = self.vocab_size

        # Flatten the image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # batch_size x num_pixels(14x14) x encoder_dim
        num_pixels = encoder_out.shape[1]

        # Embedding
        embedded_captions = self.embedding(captions)  # Batch_size x Max_Caption_lengths x embed_dim

        # LSTM
        h, c = self.init_hidden_state(encoder_out)  # Batch_size x decoder_dim

        # Do not decode as the <end> position
        decode_lengths = caption_lengths - 1

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum(
                [l > t for l in decode_lengths])  # determine training how many samples in a batch for a time step
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embedded_captions[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            pred = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = pred
            alphas[:batch_size_t, t, :] = alpha
        return predictions, captions, decode_lengths, alphas
