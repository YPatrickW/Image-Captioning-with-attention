import numpy as np
import torch
import os


def init_embedding(embeddings):  # Lecun-uniform
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


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
