import torch
from torchvision import transforms
import os
import nltk
import pickle
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from pycocotools.coco import COCO
from Build_Vocab import Vocabulary


class CocoDataset(Dataset):
    def __init__(self, root, json, vocabulary, transform=None):
        self.root = root
        self.coco_caption = COCO(json)
        self.ids = list(self.coco_caption.anns.keys())
        self.Vocal = vocabulary
        self.transform = transform

    def __getitem__(self, item):
        ann_id = self.ids[item]  # the annotation id
        image_id = self.coco_caption.anns[ann_id]["image_id"]
        image_caption = self.coco_caption.anns[ann_id]["caption"]
        path = self.coco_caption.loadImgs(image_id)[0]["file_name"]
        # Open the image as PIL
        image = Image.open(os.path.join(self.root, path)).convert("RGB")  # convert RGBA to RGB

        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.word_tokenize(image_caption.lower())

        caption_id_list = [self.Vocal("<start>")]
        caption_id_list.extend([self.Vocal[token] for token in tokens])
        caption_id_list.append(self.Vocal("<end>"))

        target = torch.tensor(caption_id_list)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(image_and_target):  # rewrite the batch iteration
    image_and_target.sort(key=lambda x: len(x[1]), reverse=True)
    images, targets = zip(*image_and_target)  # unzip the image_and_target tuple
    images = torch.stack(images, dim=0)  # Convert 3D image to 4D
    lengths = [len(target) for target in targets]
    padded_target = torch.zeros(len(targets), max(lengths)).long()
    for i, target in enumerate(targets):
        target_length = lengths[i]
        padded_target[i, :target_length] = target[:target_length]
    return images, padded_target, lengths  # the length of lengths is the batch_size


def get_loader(root, json, vocabulary, transform, batch_size, num_workers,shuffle=True):
    Coco_dataset = CocoDataset(root=root, json=json, vocabulary=vocabulary, transform=transform)
    Coco_dataloader = DataLoader(dataset=Coco_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                                 num_workers=num_workers)
    return Coco_dataloader


