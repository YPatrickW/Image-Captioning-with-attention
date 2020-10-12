import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import skimage.transform
from PIL import Image
from imageio import imread
import pickle
from Build_Vocab import *
import argparse

device = ("cuda" if torch.cuda.is_available() else "cpu")


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
    complete_seqs_scores = list()

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

        incomplete_idxs = [ind for ind, next_word in enumerate(next_word_idx) if next_word != word_dict("<end>")]
        complete_idxs = list(set(len(next_word_idx))) - set(incomplete_idxs)

        if len(complete_idxs) > 0:
            complete_seqs.extend(seqs[complete_idxs].tolist())
            complete_seqs_alpha.extend((seq_alpha[complete_idxs].tolist()))
            complete_seqs_scores.extend((top_k_scores[complete_idxs]))
            k = k - len(complete_idxs)

        if k == 0:
            break

        seqs = seqs[incomplete_idxs]
        h = h[prev_word_idx[incomplete_idxs]]
        c = c[prev_word_idx[incomplete_idxs]]
        encoder_out = encoder_out[prev_word_idx[incomplete_idxs]]
        top_k_scores = top_k_scores[incomplete_idxs].unsqueeze(1)
        k_prev_word = next_word_idx[incomplete_idxs].unsqueeze(1)

        if step > 50:
            break
        step = step + 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visualize_att(image_path, seq, alphas, word_dict, smooth=True):
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [word_dict.idx2word[idx] for idx in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cmap=plt.get_cmap("grey_r"))
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="It is time to generate captions")
    parser.add_argument("--img", "-i", help="path of image")
    parser.add_argument("--model", "-m", help="path of model")
    parser.add_argument("--word_dict", "-wd", help="path of word_dict")
    parser.add_argument("--beam_size", "-b", default=5, type=int, help="beam size for beam search")

    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location=str(device))
    decoder = checkpoint["decoder"]
    encoder = checkpoint["encoder"]
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    decoder.eval()
    encoder.eval()

    vocabulary_path = "./Vocabulary_dict/Vocab_dict.pkl"
    with open(vocabulary_path, "rb") as f:
        word_dict = pickle.load(f)
    seq, alphas = beam_search(encoder, decoder, args.img, word_dict, args.beamsize)

    alphas = torch.FloatTensor(alphas)

    visualize_att(args.img, seq, alphas, word_dict)
