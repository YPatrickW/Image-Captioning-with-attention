import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from Encoder_and_Decoder import CNN_Encoder, Decoder_with_attention
from Data_loader import *
import pickle
import os
import torch.nn as nn
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from others import *
import time


Model_cache = "./Model_cache/"

# Model parameters
embed_dim = 512
attention_dim = 512
decoder_dim = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# training parameters
start_epoch = 0
epochs = 120
epochs_since_improvement = 0
batch_size = 64
num_workers = 2
encoder_lr = 1e-4
decoder_lr = 4e-4
best_bleu4 = 0.
alpha_c = 1.
print_freq = 100
fine_tune_encoder = False
checkpoint = None


def main():
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder
    vocabulary_path = "./Vocabulary_dict/Vocab_dict.pkl"
    with open(vocabulary_path, "rb") as f:
        word_dict = pickle.load(f)
    if checkpoint is None:
        decoder = Decoder_with_attention(attention_dim=attention_dim,
                                         embed_dim=embed_dim,
                                         decoder_dim=decoder_dim,
                                         vocab_size=len(word_dict),
                                         dropout=dropout
                                         )
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.require_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = CNN_Encoder()
        encoder.requires_grad_(fine_tune_encoder)
        # if no computations for encoder's gradients, then no need for a optimizer
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.require_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        epochs_since_improvement = checkpoint["epochs_since_improvement"]
        best_bleu4 = checkpoint["bleu-4"]
        decoder = checkpoint["decoder"]
        encoder = checkpoint["encoder"]
        encoder_optimizer = checkpoint["encoder_optimizer"]
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.requires_grad_(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.require_grad, encoder.parameters()),
                                                 lr=encoder_lr)
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    Train_loader = get_loader(root="./images/resized_train/", json="./annotations/captions_train2014.json",
                              vocabulary=word_dict, transform=transforms.Compose([normalize]), batch_size=batch_size,
                              num_workers=2, shuffle=True)
    val_loader = get_loader(root="./images/resized_val/", json="./annotations/captions_val2014.json",
                            vocabulary=word_dict, transform=transforms.Compose([normalize]), batch_size=batch_size,
                            num_workers=2, shuffle=True)
    for epoch in range(start_epoch, epochs):
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_lr_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_lr_rate(encoder_optimizer, 0.8)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    encoder.train() # special mode for training: Batch_norm and dropout is True
    decoder.train()



