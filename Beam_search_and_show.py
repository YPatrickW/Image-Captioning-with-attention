import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
from PIL import Image
from imageio import imread
import pickle
from Build_Vocab import *

device = ("cuda" if torch.cuda.is_available() else "cpu")

vocabulary_path = "./Vocabulary_dict/Vocab_dict.pkl"
with open(vocabulary_path, "rb") as f:
    word_dict = pickle.load(f)


def beam_search(encoder, decoder, image_path, word_dict, beam_size=3):
    k = beam_size
    vocab_size = len(word_dict)
    img = imread(image_path)
    img = img.transpose(2, 0, 1)  # channel x width x height
    img = img / 255
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]
                                     )
    transform = transforms.Compose([normalize])
    img = transform(img)

    image = img.unsqueeze(0)  # from 3d to 4d
    encoder_out = encoder(image)
    encoded_img_dim = image.size(1)  # 14
    encoder_dim = image.size(3)  # 2048

    # Flatten image
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)

    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # a self-copy of encoder_out (3 x 196 x 2048)
    k_prev_word = torch.LongTensor([word_dict("<start>")] * k)
    k_prev_word.unsqueeze_(1).to(device)
    seqs = k_prev_word  # (k,1)

    top_k_scores = torch.zeros(k, 1).to(device)
    seq_alpha = torch.ones(k, 1, encoded_img_dim, encoded_img_dim).to(device)

    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_score = list()

    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while True:
        embedding = decoder.embedding(k_prev_word)  # batch x length x embed_dim
        # length is 1
        embedding = embedding.squeeze(1)  # batch x embed_dim
        encoding_with_attention, alpha = decoder.attention(encoder_out, h)
        alpha = alpha.view(-1, encoded_img_dim, encoded_img_dim)
        gate = decoder.sigmoid(decoder.f_beta(h))
        encoding_with_attention = gate * encoding_with_attention
        h, c = decoder.decode_step(torch.cat([embedding, encoding_with_attention], dim=1), (h, c))
        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)

        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            top_k_scores, top_k_words_idx = scores[0].topk(k, dim=0, largest=True, sorted=True)
        else:
            top_k_scores, top_k_words_idx = scores.view(-1).topk(k, dim=0, largest=True, sorted=True)

        prev_word_idx = top_k_words_idx // vocab_size
        next_word_idx = top_k_words_idx % vocab_size

        seqs = torch.cat([seqs[prev_word_idx], next_word_idx.unsqueeze(dim=1)], dim=1)
        seq_alpha = torch.cat([seq_alpha[prev_word_idx], alpha[prev_word_idx].unsqueeze(1)], dim=1)

        incomplete_idxs = [ind for ind,next_word in enumerate(next_word_idx) if next_word != word_dict("<end>")]
        complete_idxs = list()

    print(complete_seqs)


beam_search(encoder=None, decoder=None, image_path="./images/resized_train/COCO_train2014_000000000009.jpg",
            word_dict=word_dict)
