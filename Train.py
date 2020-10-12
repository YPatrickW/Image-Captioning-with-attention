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
epochs = 2
epochs_since_improvement = 0
batch_size = 64
num_workers = 2
encoder_lr = 1e-4
decoder_lr = 4e-4
grad_clip = 5
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
    vocab_size = len(word_dict)
    if checkpoint is None:
        decoder = Decoder_with_attention(attention_dim=attention_dim,
                                         embed_dim=embed_dim,
                                         decoder_dim=decoder_dim,
                                         vocab_size=vocab_size,
                                         dropout=dropout
                                         )
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = CNN_Encoder()
        encoder.requires_grad_(fine_tune_encoder)
        # if no computations for encoder's gradients, then no need for a optimizer
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
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
    To_tensor = transforms.ToTensor()

    Train_loader = get_loader(root="./images/resized_train/", json="./annotations/captions_train2014.json",
                              vocabulary=word_dict, transform=transforms.Compose([To_tensor, normalize]),
                              batch_size=batch_size,
                              num_workers=2, shuffle=True)
    val_loader = get_loader(root="./images/resized_val/", json="./annotations/captions_val2014.json",
                            vocabulary=word_dict, transform=transforms.Compose([To_tensor, normalize]),
                            batch_size=batch_size,
                            num_workers=2, shuffle=True)

    for epoch in range(start_epoch, epochs):
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_lr_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_lr_rate(encoder_optimizer, 0.8)

        train(train_loader=Train_loader, encoder=encoder, decoder=decoder, criterion=criterion,
              encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, epoch=epoch)
        bleu4 = validation(val_loader=val_loader, encoder=encoder, decoder=decoder, criterion=criterion,
                           word_dict=word_dict)

        is_best = bleu4 > best_bleu4
        best_bleu4 = max(bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement = epochs_since_improvement + 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(Model_cache, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    encoder.train()  # special mode for training: Batch_norm and dropout is True
    decoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5_accuracy = AverageMeter()  # top5 accuracy

    start = time.time()

    for i, (images, captions, caption_lengths) in enumerate(train_loader):
        data_time.update(time.time() - start)
        images = images.to(device)
        captions = captions.to(device)
        caption_lengths = torch.Tensor(caption_lengths).to(device)

        images = encoder(images)

        scores, captions, decoded_lengths, alphas = decoder(images, captions, caption_lengths)

        targets = captions[:, 1:]

        decoded_lengths_tensor = torch.Tensor(decoded_lengths)

        scores = pack_padded_sequence(scores, decoded_lengths_tensor, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decoded_lengths_tensor, batch_first=True)[0]

        loss = criterion(scores, targets)

        loss = loss + alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decoded_lengths))
        top5_accuracy.update(top5, sum(decoded_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print("[{0}][{1}/{2}]\t".format(epoch, i, len(train_loader)),
                  f"Batch Time {batch_time.val:.3f}({batch_time.avg:.3f})\t".format(batch_time=batch_time),
                  f"Data load Time {data_time.val:.3f}({data_time.avg:.3f})\t".format(data_time=data_time),
                  f"Loss {losses.val:.4f}({losses.avg:.4f})\t".format(losses=losses),
                  f"Top 5 Accuracy{top5_accuracy.val:.3f}({top5_accuracy.avg:.3f})\t".format(top5_accuracy=top5_accuracy)
                  )


def validation(val_loader, encoder, decoder, criterion, word_dict):
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5_accuracy = AverageMeter()

    start = time.time()

    true_caption = list()
    prediction = list()

    with torch.no_grad():
        for i, (images, captions, caption_lengths) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths(device)

            if encoder is not None:
                images = encoder(images)
            scores, captions, decoded_lengths, alphas = decoder(images, captions, caption_lengths)

            targets = captions[:, 1:]
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decoded_lengths, batch_first=True)
            target, _ = pack_padded_sequence(targets, decoded_lengths, batch_first=True)

            loss = criterion(scores, target)
            loss = loss + alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decoded_lengths))
            top5_accuracy.update(top5, sum(decoded_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print("[{0}/{1}]\t".format(i, len(val_loader)),
                      f"Batch Time {batch_time.val:.3f}({batch_time.avg:.3f})\t".format(batch_time=batch_time),
                      f"Loss {loss.val:.4f}({loss.avg:.4f})\t".format(loss=losses),
                      f"Top 5 Accuracy{top5.val:.3f}({top5.avg:.3f})\t".format(top5=top5_accuracy)
                      )
            for j in range(captions.shape[0]):
                single_caption = captions[j].tolist()
                image_caption = [word for word in single_caption if
                                 word not in {word_dict("<start>"), word_dict("<pad>")}]
                true_caption.append(image_caption)

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            single_pred_list = []
            for j, p in enumerate(preds):
                single_pred_list.append(preds[j][:decoded_lengths[j]])
            preds = single_pred_list
            prediction.extend(preds)

            assert len(true_caption) == len(prediction)

        bleu4 = corpus_bleu(true_caption, prediction)

        print(f"Loss:{loss.avg:.3f}\t".format(loss=losses),
              f"Top-5 Accuracy:{top5.avg:.3f}\t".format(top5=top5_accuracy),
              f"bleu:{bleu4}".format(bleu4=bleu4)
              )
        return bleu4


if __name__ == '__main__':
    main()
