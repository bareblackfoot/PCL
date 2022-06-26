from PIL import ImageFilter
import random
import torchvision.datasets as datasets
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    

class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index

class HabitatImageDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def augment(self, img):
        data_patches = np.stack([img[:, i * 21:(i + 1) * 21] for i in range(12)])
        index_list = np.arange(0, 12).tolist()
        random_cut = np.random.randint(12)
        index_list = index_list[random_cut:] + index_list[:random_cut]
        permuted_patches = data_patches[index_list]
        augmented_img = np.concatenate(np.split(permuted_patches, 12, axis=0), 2)[0]
        return augmented_img

    def pull_image(self, index):
        x = plt.imread(self.data_list[index])
        # im = Image.fromarray(np.uint8(x[...,:3] * 255))
        x_aug = self.augment(x)
        q = x[...,:3]
        k = x_aug[...,:3] #Image.fromarray(np.uint8(x_aug[...,:3] * 255))
        # q = self.base_transform(im)
        # k = self.base_transform(augmented_img)
        if self.noisydepth:
            q = torch.cat([torch.tensor(q).permute(2,0,1), torch.tensor(x[...,-1:]).permute(2,0,1)], 0)
            k = torch.cat([torch.tensor(k).permute(2,0,1), torch.tensor(x_aug[...,-1:]).permute(2,0,1)], 0)
        return [q, k], index

class HabitatImageEvalDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def pull_image(self, index):
        x = plt.imread(self.data_list[index])
        q = x[...,:3]
        # im = Image.fromarray(np.uint8(x[...,:3] * 255))
        # q = self.base_transform(im)
        if self.noisydepth:
            q = torch.cat([torch.tensor(q).permute(2,0,1), torch.tensor(x[...,-1:]).permute(2,0,1)], 0)
            # q = torch.cat([q, torch.tensor(x[...,-1:]).permute(2,0,1)], 0)
        return q, index
