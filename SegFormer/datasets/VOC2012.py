import torch
import numpy as np
import os
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
import PIL
import torchvision
from torchvision import transforms

        
def color_map(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = []
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap.append([r, g, b])

    return cmap

class VOC2012(torchvision.datasets.VOCSegmentation):
    # dataset: Dataset
    # """
    # num_classes: 19
    # """

    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
    ]

    # PALETTE = torch.tensor([
    #     [0, 0, 0],
    #     [128, 0, 0],
    #     [0, 128, 0],
    #     [128, 128, 0],
    #     [0, 0, 128],
    #     [128, 0, 128],
    #     [0, 128, 128],
    #     [128, 128, 128],
    #     [64, 0, 0],
    #     [192, 0, 0],
    #     [64, 128, 0],
    #     [192, 128, 0],
    #     [64, 0, 128],
    #     [192, 0, 128],
    #     [64, 128, 128],
    #     [192, 128, 128],
    #     [0, 64, 0],
    #     [128, 64, 0],
    #     [0, 192, 0],
    #     [128, 192, 0],
    #     [0, 64, 128],
    # ])

    def __init__(self, root: str, split: str = 'train', transform=None) -> None:
        assert split in ['train', 'val', 'test']

        self.ignore_label = 255
        self.PALETTE = torch.tensor(color_map())
        self.n_classes = len(self.CLASSES)

        super().__init__(root=root, image_set=split, download=True,
                         transform=transforms.Compose([transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR), transforms.ToTensor()]),
                         target_transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256), interpolation=PIL.Image.NEAREST)]))


    def _convert_to_segmentation_mask(self, mask):
        mask = mask.transpose(0, 2)
        mask = (mask * 255).long()
        height, width = mask.shape[:2]
        segmentation_mask = torch.zeros((height, width)).long()

        # for i in range(height):
        #     for j in range(width):
        #         print(mask[i, j, :])
        
        # exit(0)

        for label_index, label in enumerate(self.PALETTE):
            # print("mask: ", segmentation_mask.shape, mask.shape, label)
            temp_segmentation_mask = torch.zeros((height, width, 3)).bool()
            for i in range(3):
                temp_segmentation_mask[:, :, i][mask[:, :, i] == label[i]] = True
            
            temp_segmentation_mask = temp_segmentation_mask.all(dim=-1)
            segmentation_mask[temp_segmentation_mask] = label_index

        # for i in range(height):
        #     for j in range(width):
        #         print(segmentation_mask[i, j])
        
        return segmentation_mask.transpose(0, 1)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("RGB")

        # print(self.masks[index])

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        
        # print("1st", image.shape, mask.shape)
        mask = self._convert_to_segmentation_mask(mask)
        # print("2nd", image.shape, mask.shape)
        return image, mask


    # def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
    #     result1, result2 = self.dataset.__getitem__(index)
    #     # print(result2.squeeze())
    #     return result1, result2.squeeze().long()
