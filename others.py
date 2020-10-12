import numpy as np
import torch
import os


def adjust_lr_rate(optimizer, shrink_factor):
    print("It is time to decay the learning rate")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    print("The new learning rate is %f" % (optimizer.param_groups[0]["lr"]))


def save_checkpoint(Model_cache, epoch, epoch_since_improvement, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, blue4, is_best, Data_name="Caption_with_attention"):
    state = {"epoch": epoch,
             "epoch_since_improvement": epoch_since_improvement,
             "encoder": encoder,
             "decoder": decoder,
             "encoder_optimizer": encoder_optimizer,
             "decoder_optimizer": decoder_optimizer
             }
    filename = "checkpoint_" + Data_name + ".pth.tar"
    filepath = os.path.join(Model_cache, filename)
    torch.save(state, filepath)
    if is_best:
        filename_best = "best_checkpoint_" + Data_name + ".pth.tar"
        filepath_best = os.path.join(Model_cache, filename_best)
        torch.save(state, filepath_best)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
