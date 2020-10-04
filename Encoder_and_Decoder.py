import torch
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CNN_Encoder(torch.nn.Module):
    def __init__(self,embed_size=14):
        super(CNN_Encoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2] # delete the last FC and pool layer
        self.resnet = torch.nn.Sequential(*modules) # * is used to split the generator to single element
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((embed_size,embed_size))
    def forward(self,images):
        # images: batch_size x 3 x width x height (width,height is 256)
        # out : batch_size x 2048 x width/3 x height/3 (out is 8)
        out = self.resnet(images)
        out = self.adaptive_pool(out) # transform the out's width and height to 14
        out = out.permute(0,2,3,1) # batch_size x embed_size x embed_size x channel(2048)
        return out








