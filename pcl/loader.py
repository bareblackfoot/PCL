from PIL import ImageFilter
import random
import torchvision.datasets as datasets
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torch
import joblib, glob, os
import cv2
from PIL import Image
with open(os.path.join(os.path.dirname(__file__), "data/coco_category.txt"), "r") as f:
    lines = f.readlines()
DETECTION_CATEGORIES = [line.rstrip() for line in lines]
COCO_CATEGORIES = DETECTION_CATEGORIES
# 40 category of interests
with open(os.path.join(os.path.dirname(__file__), "data/matterport_category.txt"), "r") as f:
    lines = f.readlines()
CATEGORIES = {}
CATEGORIES['mp3d'] = [line.rstrip() for line in lines]
CATEGORIES['hm3d'] = [line.rstrip() for line in lines]
CATEGORIES['gibson'] = DETECTION_CATEGORIES

d3_40_colors_rgb: np.ndarray = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)


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


class HabitatVideoDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, temporal_transform_q=None, temporal_transform_k=None, sample_duration=16, maximum_duration=100):
        self.data_list = data_list
        self.base_transform = base_transform
        self.temporal_transform_q = temporal_transform_q
        self.temporal_transform_k = temporal_transform_k
        self.sample_duration = sample_duration
        self.maximum_duration = maximum_duration
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-2] for i in range(len(self.data_list))]))

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def pull_image(self, index):
        imgs = os.listdir(self.data_list[index])
        scene = self.data_list[index].split("/")[-2]
        scene_idx = self.scenes.index(scene)
        imgs = [img for img in imgs if ".jpg" in img]
        imgs = sorted(imgs)
        frame_indices = list(np.arange(len(imgs)))
        if self.temporal_transform_q is not None:
            frame_indices, mask_q = self.temporal_transform_q(frame_indices)
        # frame_indices_ = frame_indices[-self.sample_duration:]
        imgs_ = [imgs[idx] for idx in frame_indices]
        x = []
        for img_path in imgs_:
            img = plt.imread(os.path.join(self.data_list[index], img_path))
            img = self.base_transform(Image.fromarray(img))
            x.append(img)
        q = torch.stack(x)

        frame_indices = list(np.arange(len(imgs)))
        if self.temporal_transform_k is not None:
            frame_indices, mask_k = self.temporal_transform_k(frame_indices)
        imgs_ = [imgs[idx] for idx in frame_indices]
        x = []
        for img_path in imgs_:
            img = plt.imread(os.path.join(self.data_list[index], img_path))
            img = self.base_transform(Image.fromarray(img))
            x.append(img)
        k = torch.stack(x)
        mask_q = torch.tensor(mask_q).float()
        mask_k = torch.tensor(mask_k).float()
        return [q, k], [mask_q, mask_k], index, scene_idx


    # def pull_image(self, index):
    #     imgs = os.listdir(self.data_list[index])
    #     imgs = [img for img in imgs if ".jpg" in img]
    #     imgs = sorted(imgs)
    #     frame_indices = list(np.arange(len(imgs)))
    #     if self.temporal_transform_q is not None:
    #         frame_indices = self.temporal_transform_q(frame_indices)
    #     # frame_indices_ = frame_indices[-self.sample_duration:]
    #     imgs_ = [imgs[idx] for idx in frame_indices]
    #     x = []
    #     for img_path in imgs_:
    #         img = plt.imread(os.path.join(self.data_list[index], img_path))
    #         x.append(img)
    #     x = np.stack(x)
    #     q = torch.tensor(x).permute(3, 0, 1, 2).float()
    #
    #     frame_indices = list(np.arange(len(imgs)))
    #     if self.temporal_transform_k is not None:
    #         frame_indices = self.temporal_transform_k(frame_indices)
    #     imgs_ = [imgs[idx] for idx in frame_indices]
    #     x = []
    #     for img_path in imgs_:
    #         img = plt.imread(os.path.join(self.data_list[index], img_path))
    #         x.append(img)
    #     x = np.stack(x)
    #     k = torch.tensor(x).permute(3, 0, 1, 2).float()
    #     return [q, k], index


class HabitatVideoEvalDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, temporal_transform=None, sample_duration=16):
        self.data_list = data_list
        self.base_transform = base_transform
        self.temporal_transform = temporal_transform
        self.sample_duration = sample_duration
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-2] for i in range(len(self.data_list))]))

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def pull_image(self, index):
        scene = self.data_list[index].split("/")[-2]
        scene_idx = self.scenes.index(scene)
        imgs = os.listdir(self.data_list[index])
        imgs = [img for img in imgs if ".jpg" in img]
        imgs = sorted(imgs)
        frame_indices = list(np.arange(len(imgs)))
        if self.temporal_transform is not None:
            frame_indices, mask_q = self.temporal_transform(frame_indices)
        # frame_indices_ = frame_indices[-self.sample_duration:]
        imgs_ = [imgs[idx] for idx in frame_indices]
        x = []
        for img_path in imgs_:
            img = plt.imread(os.path.join(self.data_list[index], img_path))
            img = self.base_transform(Image.fromarray(img))
            x.append(img)
        q = torch.stack(x)
        mask_q = torch.tensor(mask_q).float()
        return q, mask_q, index, scene_idx

    # def pull_image(self, index):
    #     imgs = os.listdir(self.data_list[index])
    #     imgs = [img for img in imgs if ".jpg" in img]
    #     imgs = sorted(imgs)
    #     frame_indices = list(np.arange(len(imgs)))
    #     if self.temporal_transform is not None:
    #         frame_indices = self.temporal_transform(frame_indices)
    #     frame_indices_ = frame_indices[-self.sample_duration:]
    #     imgs_ = [imgs[idx] for idx in frame_indices_]
    #     x = []
    #     for img_path in imgs_:
    #         img = plt.imread(os.path.join(self.data_list[index], img_path))
    #         x.append(img)
    #     x = np.stack(x)
    #     q = torch.tensor(x).permute(3, 0, 1, 2).float()
    #     return q, index



class HabitatImageDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-3] for i in range(len(self.data_list))]))
        # self.places = sorted(np.unique([self.data_list[i].split("/")[-2] for i in range(len(self.data_list))]))
        self.places = ["kitchen", "bedroom", "bathroom","closet", "living room"]
        # data_dir = "/".join(self.data_list[0].split("/")[:-3]).replace("val", "train")
        # scenes = os.listdir(data_dir)
        # places = []
        # for scene in scenes:
        #     places.append(glob.glob(os.path.join(data_dir, scene) + "/*"))
        # self.places = sorted(np.unique(places))

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
        scene = self.data_list[index].split("/")[-3]
        scene_idx = self.scenes.index(scene)
        place = self.data_list[index].split("/")[-2]
        place_idx = self.places.index(place)
        scene_data_list = [self.data_list[i] for i in range(len(self.data_list)) if self.data_list[i].split("/")[-3] == scene]
        scene_data_list.remove(self.data_list[index])
        idx = np.random.randint(len(scene_data_list))
        negative_sample = scene_data_list[idx]
        n = plt.imread(negative_sample)
        n = torch.tensor(n[...,:3]).permute(2,0,1)

        x = plt.imread(self.data_list[index])
        x_aug = self.augment(x)
        if self.base_transform is not None:
            q = self.base_transform(Image.fromarray((x[...,:3]*255.).astype(np.uint8)))
            k = self.base_transform(Image.fromarray((x_aug[...,:3]*255.).astype(np.uint8)))
        else:
            q = torch.tensor(x[...,:3]).permute(2,0,1)
            k = torch.tensor(x_aug[...,:3]).permute(2,0,1)
        if self.noisydepth:
            q = torch.cat([q, torch.tensor(x[...,-1:]).permute(2,0,1)], 0)
            k = torch.cat([k, torch.tensor(x_aug[...,-1:]).permute(2,0,1)], 0)
        return [q, k, n], index, scene_idx, place_idx

class HabitatImageEvalDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-3] for i in range(len(self.data_list))]))
        # self.places = sorted(np.unique([self.data_list[i].split("/")[-2] for i in range(len(self.data_list))]))
        # data_dir = "/".join(self.data_list[0].split("/")[:-3]).replace("val", "train")
        # scenes = os.listdir(data_dir)
        # places = []
        # for scene in scenes:
        #     places.append(glob.glob(os.path.join(data_dir, scene) + "/*"))
        # places = np.concatenate(places)
        # self.places = sorted(np.unique([places[i].split("/")[-1] for i in range(len(places))]))
        self.places = ["kitchen", "bedroom", "bathroom","closet", "living room"]

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def pull_image(self, index):
        scene = self.data_list[index].split("/")[-3]
        scene_idx = self.scenes.index(scene)
        place = self.data_list[index].split("/")[-2]
        place_idx = self.places.index(place)
        x = plt.imread(self.data_list[index])
        if self.base_transform is not None:
            q = self.base_transform(Image.fromarray((x[...,:3]*255.).astype(np.uint8)))
        else:
            q = torch.tensor(x[...,:3]).permute(2,0,1)
        if self.noisydepth:
            q = torch.cat([q, torch.tensor(x[...,-1:]).permute(2,0,1)], 0)
        return q, index, scene_idx, place_idx


class HabitatImageSemDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-3] for i in range(len(self.data_list))]))
        self.places = sorted(np.unique([self.data_list[i].split("/")[-2] for i in range(len(self.data_list))]))

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def augment(self, img, sem):
        data_patches = np.stack([img[:, i * 21:(i + 1) * 21] for i in range(12)])
        sem_data_patches = np.stack([sem[:, i * 21:(i + 1) * 21] for i in range(12)])
        index_list = np.arange(0, 12).tolist()
        random_cut = np.random.randint(12)
        index_list = index_list[random_cut:] + index_list[:random_cut]
        permuted_patches = data_patches[index_list]
        permuted_sem_patches = sem_data_patches[index_list]
        augmented_img = np.concatenate(np.split(permuted_patches, 12, axis=0), 2)[0]
        augmented_sem = np.concatenate(np.split(permuted_sem_patches, 12, axis=0), 2)[0]
        return augmented_img, augmented_sem

    def pull_image(self, index):
        scene = self.data_list[index].split("/")[-3]
        scene_idx = self.scenes.index(scene)
        place = self.data_list[index].split("/")[-2]
        place_idx = self.places.index(place)

        same_place_list = [self.data_list[i] for i in range(len(self.data_list)) if self.data_list[i].split("/")[-2] == place]
        if len(same_place_list) > 1:
            same_place_list.remove(self.data_list[index])
        idx = np.random.randint(len(same_place_list))
        sp_sample = same_place_list[idx]
        sp = plt.imread(sp_sample)
        sp_sem_ = cv2.imread(sp_sample.replace("rgb", "sem"))[...,0:1]
        sp_sem = Image.new(
            "P", (sp_sem_.shape[1], sp_sem_.shape[0])
        )
        sp_sem.putpalette(d3_40_colors_rgb.flatten())
        sp_sem.putdata((sp_sem_.flatten() % 40).astype(np.uint8))
        sp_sem = np.array(sp_sem.convert("RGBA"))[...,:3]/255.
        # sp = torch.tensor(np.concatenate([sp[...,:3], semantic_img], -1)).permute(2,0,1).float()

        scene_data_list = [self.data_list[i] for i in range(len(self.data_list)) if self.data_list[i].split("/")[-3] == scene]
        scene_data_list.remove(self.data_list[index])
        idx = np.random.randint(len(scene_data_list))
        negative_sample = scene_data_list[idx]
        n = plt.imread(negative_sample)
        n_sem_ = cv2.imread(negative_sample.replace("rgb", "sem"))[...,0:1]
        n_sem = Image.new(
            "P", (n_sem_.shape[1], n_sem_.shape[0])
        )
        n_sem.putpalette(d3_40_colors_rgb.flatten())
        n_sem.putdata((n_sem_.flatten() % 40).astype(np.uint8))
        n_sem = np.array(n_sem.convert("RGBA"))[...,:3]/255.
        # n = torch.tensor(np.concatenate([n[...,:3], semantic_img], -1)).permute(2,0,1).float()

        x = plt.imread(self.data_list[index])

        x_sem = cv2.imread(self.data_list[index].replace("rgb", "sem"))[...,0:1]
        semantic_img = Image.new(
            "P", (x_sem.shape[1], x_sem.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((x_sem.flatten() % 40).astype(np.uint8))
        semantic_img = np.array(semantic_img.convert("RGBA"))[...,:3]/255.
        x_aug, x_aug_sem = self.augment(x, semantic_img)
        if self.base_transform is not None:
            q_rgb = self.base_transform(Image.fromarray((x[...,:3]*255.).astype(np.uint8)))
            q_sem = self.base_transform(Image.fromarray((semantic_img*255.).astype(np.uint8)))
            k = self.base_transform(Image.fromarray((x_aug[...,:3]*255.).astype(np.uint8)))
            k_sem = self.base_transform(Image.fromarray((x_aug_sem*255.).astype(np.uint8)))
            n = self.base_transform(Image.fromarray((n[...,:3]*255.).astype(np.uint8)))
            n_sem = self.base_transform(Image.fromarray((n_sem*255.).astype(np.uint8)))
            sp = self.base_transform(Image.fromarray((sp[...,:3]*255.).astype(np.uint8)))
            sp_sem = self.base_transform(Image.fromarray((sp_sem*255.).astype(np.uint8)))
            q = torch.cat([q_rgb, q_sem], 0)
            k = torch.cat([k, k_sem], 0)
            n = torch.cat([n, n_sem], 0)
            sp = torch.cat([sp, sp_sem], 0)
        else:
            # q = torch.tensor(x[...,:3]).permute(2,0,1)
            # k = torch.tensor(x_aug[...,:3]).permute(2,0,1)
            q = torch.tensor(np.concatenate([x[...,:3], semantic_img], -1)).permute(2,0,1).float()
            k = torch.tensor(np.concatenate([x_aug[...,:3], x_aug_sem], -1)).permute(2,0,1).float()
            n = torch.tensor(np.concatenate([n[...,:3], n_sem], -1)).permute(2,0,1).float()
            sp = torch.tensor(np.concatenate([sp[...,:3], sp_sem], -1)).permute(2,0,1).float()
        return [q, k, n, sp], index, scene_idx, place_idx


class HabitatImageSemEvalDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-2] for i in range(len(self.data_list))]))

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def pull_image(self, index):
        scene = self.data_list[index].split("/")[-2]
        scene_idx = self.scenes.index(scene)
        x = plt.imread(self.data_list[index])
        x_sem = cv2.imread(self.data_list[index].replace("rgb", "sem"))[...,0:1]
        semantic_img = Image.new(
            "P", (x_sem.shape[1], x_sem.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((x_sem.flatten() % 40).astype(np.uint8))
        semantic_img = np.array(semantic_img.convert("RGBA"))[...,:3]/255.
        if self.base_transform is not None:
            q_rgb = self.base_transform(Image.fromarray((x[...,:3]*255.).astype(np.uint8)))
            q_sem = self.base_transform(Image.fromarray((semantic_img*255.).astype(np.uint8)))
            q = torch.cat([q_rgb, q_sem], 0)
        else:
            q = torch.tensor(np.concatenate([x[...,:3], semantic_img], -1)).permute(2,0,1).float()
        return q, index, scene_idx

#
# class HabitatImageSemDataset(data.Dataset):
#     def __init__(self, data_list, base_transform=None, noisydepth=False):
#         self.data_list = data_list
#         self.base_transform = base_transform
#         self.noisydepth = noisydepth
#         self.scenes = sorted(np.unique([self.data_list[i].split("/")[-3] for i in range(len(self.data_list))]))
#         self.places = sorted(np.unique([self.data_list[i].split("/")[-2] for i in range(len(self.data_list))]))
#
#     def __getitem__(self, index):
#         return self.pull_image(index)
#
#     def __len__(self):
#         return len(self.data_list)
#
#     def augment(self, img, sem):
#         data_patches = np.stack([img[:, i * 21:(i + 1) * 21] for i in range(12)])
#         sem_data_patches = np.stack([sem[:, i * 21:(i + 1) * 21] for i in range(12)])
#         index_list = np.arange(0, 12).tolist()
#         random_cut = np.random.randint(12)
#         index_list = index_list[random_cut:] + index_list[:random_cut]
#         permuted_patches = data_patches[index_list]
#         permuted_sem_patches = sem_data_patches[index_list]
#         augmented_img = np.concatenate(np.split(permuted_patches, 12, axis=0), 2)[0]
#         augmented_sem = np.concatenate(np.split(permuted_sem_patches, 12, axis=0), 2)[0]
#         return augmented_img, augmented_sem
#
#     def pull_image(self, index):
#         scene = self.data_list[index].split("/")[-3]
#         scene_idx = self.scenes.index(scene)
#         place = self.data_list[index].split("/")[-2]
#         place_idx = self.places.index(place)
#
#         same_place_list = [self.data_list[i] for i in range(len(self.data_list)) if self.data_list[i].split("/")[-2] == place]
#         if len(same_place_list) > 1:
#             same_place_list.remove(self.data_list[index])
#         idx = np.random.randint(len(same_place_list))
#         sp_sample = same_place_list[idx]
#         sp = plt.imread(sp_sample)
#         sp_sem_ = cv2.imread(sp_sample.replace("rgb", "sem"))[...,0:1]
#         sp_sem = Image.new(
#             "P", (sp_sem_.shape[1], sp_sem_.shape[0])
#         )
#         sp_sem.putpalette(d3_40_colors_rgb.flatten())
#         sp_sem.putdata((sp_sem_.flatten() % 40).astype(np.uint8))
#         sp_sem = np.array(sp_sem.convert("RGBA"))[...,:3]/255.
#         # sp = torch.tensor(np.concatenate([sp[...,:3], semantic_img], -1)).permute(2,0,1).float()
#
#         scene_data_list = [self.data_list[i] for i in range(len(self.data_list)) if self.data_list[i].split("/")[-3] == scene]
#         scene_data_list.remove(self.data_list[index])
#         idx = np.random.randint(len(scene_data_list))
#         negative_sample = scene_data_list[idx]
#         n = plt.imread(negative_sample)
#         n_sem_ = cv2.imread(negative_sample.replace("rgb", "sem"))[...,0:1]
#         n_sem = Image.new(
#             "P", (n_sem_.shape[1], n_sem_.shape[0])
#         )
#         # from habitat_sim.utils.common import d3_40_colors_rgb
#         n_sem.putpalette(d3_40_colors_rgb.flatten())
#         n_sem.putdata((n_sem_.flatten() % 40).astype(np.uint8))
#         n_sem = np.array(n_sem.convert("RGBA"))[...,:3]/255.
#         # n = torch.tensor(np.concatenate([n[...,:3], semantic_img], -1)).permute(2,0,1).float()
#
#         x = plt.imread(self.data_list[index])
#
#         x_sem = cv2.imread(self.data_list[index].replace("rgb", "sem"))[...,0:1]
#         semantic_img = Image.new(
#             "P", (x_sem.shape[1], x_sem.shape[0])
#         )
#         semantic_img.putpalette(d3_40_colors_rgb.flatten())
#         semantic_img.putdata((x_sem.flatten() % 40).astype(np.uint8))
#         semantic_img = np.array(semantic_img.convert("RGBA"))[...,:3]/255.
#         x_aug, x_aug_sem = self.augment(x, semantic_img)
#         if self.base_transform is not None:
#             q_rgb = self.base_transform(Image.fromarray((x[...,:3]*255.).astype(np.uint8)))
#             q_sem = self.base_transform(Image.fromarray((semantic_img*255.).astype(np.uint8)))
#             k = self.base_transform(Image.fromarray((x_aug[...,:3]*255.).astype(np.uint8)))
#             k_sem = self.base_transform(Image.fromarray((x_aug_sem*255.).astype(np.uint8)))
#             n = self.base_transform(Image.fromarray((n[...,:3]*255.).astype(np.uint8)))
#             n_sem = self.base_transform(Image.fromarray((n_sem*255.).astype(np.uint8)))
#             sp = self.base_transform(Image.fromarray((sp[...,:3]*255.).astype(np.uint8)))
#             sp_sem = self.base_transform(Image.fromarray((sp_sem*255.).astype(np.uint8)))
#             q = torch.cat([q_rgb, q_sem], 0)
#             k = torch.cat([k, k_sem], 0)
#             n = torch.cat([n, n_sem], 0)
#             sp = torch.cat([sp, sp_sem], 0)
#         else:
#             # q = torch.tensor(x[...,:3]).permute(2,0,1)
#             # k = torch.tensor(x_aug[...,:3]).permute(2,0,1)
#             q = torch.tensor(np.concatenate([x[...,:3], semantic_img], -1)).permute(2,0,1).float()
#             k = torch.tensor(np.concatenate([x_aug[...,:3], x_aug_sem], -1)).permute(2,0,1).float()
#             n = torch.tensor(np.concatenate([n[...,:3], n_sem], -1)).permute(2,0,1).float()
#             sp = torch.tensor(np.concatenate([sp[...,:3], sp_sem], -1)).permute(2,0,1).float()
#         return [q, k, n, sp], index, scene_idx, place_idx


class HabitatSemDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-3] for i in range(len(self.data_list))]))
        COI = ['chair', 'door', 'table', 'picture', 'cabinet', 'cushion', 'window', 'sofa',
                   'bed', 'curtain', 'chest_of_drawers', 'plant', 'sink', 'stairs',  'toilet', 'stool', 'towel',
                   'mirror', 'tv_monitor', 'shower', 'bathtub', 'counter', 'fireplace',
                   'shelving', 'blinds', 'gym_equipment', 'seating', 'board', 'furniture', 'appliances', 'clothes', 'objects']
        self.COI_INDEX = np.where([c in COI for c in CATEGORIES['mp3d']])[0]
        self.d3_40_colors_rgb = d3_40_colors_rgb.copy()
        for idx in range(40):
            if idx not in self.COI_INDEX:
                self.d3_40_colors_rgb[idx] = np.array([0,0,0])

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def augment(self, sem):
        sem_data_patches = np.stack([sem[:, i * 21:(i + 1) * 21] for i in range(12)])
        index_list = np.arange(0, 12).tolist()
        random_cut = np.random.randint(12)
        index_list = index_list[random_cut:] + index_list[:random_cut]
        permuted_sem_patches = sem_data_patches[index_list]
        augmented_sem = np.concatenate(np.split(permuted_sem_patches, 12, axis=0), 2)[0]
        return augmented_sem

    def pull_image(self, index):
        x_sem = cv2.imread(self.data_list[index].replace("rgb", "sem"))[...,0:1]
        semantic_img = Image.new(
            "P", (x_sem.shape[1], x_sem.shape[0])
        )
        semantic_img.putpalette(self.d3_40_colors_rgb.flatten())
        semantic_img.putdata((x_sem.flatten()).astype(np.uint8))
        semantic_img = np.array(semantic_img.convert("RGBA"))[...,:3]/255.

        place = self.data_list[index].split("/")[-2]
        same_place_list = [self.data_list[i] for i in range(len(self.data_list)) if self.data_list[i].split("/")[-2] == place]
        if len(same_place_list) > 1:
            same_place_list.remove(self.data_list[index])
        idx = np.random.randint(len(same_place_list))
        sp_sample = same_place_list[idx]
        # sp = plt.imread(sp_sample)
        sp_sem_ = cv2.imread(sp_sample.replace("rgb", "sem"))[...,0:1]
        sp_sem = Image.new(
            "P", (sp_sem_.shape[1], sp_sem_.shape[0])
        )
        sp_sem.putpalette(self.d3_40_colors_rgb.flatten())
        sp_sem.putdata((sp_sem_.flatten() % 40).astype(np.uint8))
        sp_sem = np.array(sp_sem.convert("RGBA"))[...,:3]/255.
        x_aug_sem = self.augment(sp_sem)

        if self.base_transform is not None:
            q = self.base_transform(Image.fromarray((semantic_img*255.).astype(np.uint8)))
            k1 = self.base_transform(Image.fromarray((x_aug_sem*255.).astype(np.uint8)))
            k2 = self.base_transform(Image.fromarray((sp_sem*255.).astype(np.uint8)))
            # n = self.base_transform(Image.fromarray((n_sem_img*255.).astype(np.uint8)))
        else:
            q = torch.tensor(semantic_img).permute(2,0,1).float()
            k1 = torch.tensor(x_aug_sem).permute(2,0,1).float()
            k2 = torch.tensor(sp_sem).permute(2,0,1).float()
            # n = torch.tensor(n_sem_img).permute(2,0,1).float()
        return [q, k1, k2], index


class HabitatSemEvalDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-2] for i in range(len(self.data_list))]))

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def pull_image(self, index):
        scene = self.data_list[index].split("/")[-2]
        scene_idx = self.scenes.index(scene)
        x_sem = cv2.imread(self.data_list[index].replace("rgb", "sem"))[...,0:1]
        semantic_img = Image.new(
            "P", (x_sem.shape[1], x_sem.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((x_sem.flatten() % 40).astype(np.uint8))
        semantic_img = np.array(semantic_img.convert("RGBA"))[...,:3]/255.
        q = torch.tensor(semantic_img).permute(2,0,1).float()
        return q, index, scene_idx


class HabitatSemObjwiseDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-3] for i in range(len(self.data_list))]))
        objects = os.listdir("/".join(self.data_list[0].split("/")[:-2]))
        self.same_place_dict = {}
        for object in objects:
            self.same_place_dict[object] = [self.data_list[i] for i in range(len(self.data_list)) if self.data_list[i].split("/")[-2] == object]

        COI = ['chair', 'door', 'table', 'picture', 'cabinet', 'cushion', 'window', 'sofa',
                   'bed', 'curtain', 'chest_of_drawers', 'plant', 'sink', 'stairs',  'toilet', 'stool', 'towel',
                   'mirror', 'tv_monitor', 'shower', 'bathtub', 'counter', 'fireplace',
                   'shelving', 'blinds', 'gym_equipment', 'seating', 'board', 'furniture', 'appliances', 'clothes', 'objects']
        self.COI_INDEX = np.where([c in COI for c in CATEGORIES['mp3d']])[0]
        self.d3_40_colors_rgb = d3_40_colors_rgb.copy()
        for idx in range(40):
            if idx not in self.COI_INDEX:
                self.d3_40_colors_rgb[idx] = np.array([0,0,0])

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def augment(self, sem):
        sem_data_patches = np.stack([sem[:, i * 21:(i + 1) * 21] for i in range(12)])
        index_list = np.arange(0, 12).tolist()
        random_cut = np.random.randint(12)
        index_list = index_list[random_cut:] + index_list[:random_cut]
        permuted_sem_patches = sem_data_patches[index_list]
        augmented_sem = np.concatenate(np.split(permuted_sem_patches, 12, axis=0), 2)[0]
        return augmented_sem

    def pull_image(self, index):
        x_sem = cv2.imread(self.data_list[index].replace("rgb", "sem"))[...,0:1]
        semantic_img = Image.new(
            "P", (x_sem.shape[1], x_sem.shape[0])
        )
        semantic_img.putpalette(self.d3_40_colors_rgb.flatten())
        semantic_img.putdata((x_sem.flatten()).astype(np.uint8))
        semantic_img = np.array(semantic_img.convert("RGBA"))[...,:3]/255.
        semantic_img = self.augment(semantic_img)

        common_obj = self.data_list[index].split("/")[-2]
        same_place_list = self.same_place_dict[common_obj].copy()
        if len(same_place_list) > 1:
            same_place_list.remove(self.data_list[index])
        idx = np.random.randint(len(same_place_list))
        sp_sample = same_place_list[idx]
        # sp = plt.imread(sp_sample)
        sp_sem_ = cv2.imread(sp_sample.replace("rgb", "sem"))[...,0:1]
        sp_sem = Image.new(
            "P", (sp_sem_.shape[1], sp_sem_.shape[0])
        )
        sp_sem.putpalette(self.d3_40_colors_rgb.flatten())
        sp_sem.putdata((sp_sem_.flatten() % 40).astype(np.uint8))
        sp_sem = np.array(sp_sem.convert("RGBA"))[...,:3]/255.
        x_aug_sem = self.augment(sp_sem)

        if self.base_transform is not None:
            q = self.base_transform(Image.fromarray((semantic_img*255.).astype(np.uint8)))
            k1 = self.base_transform(Image.fromarray((x_aug_sem*255.).astype(np.uint8)))
            k2 = self.base_transform(Image.fromarray((sp_sem*255.).astype(np.uint8)))
            # n = self.base_transform(Image.fromarray((n_sem_img*255.).astype(np.uint8)))
        else:
            q = torch.tensor(semantic_img).permute(2,0,1).float()
            k1 = torch.tensor(x_aug_sem).permute(2,0,1).float()
            k2 = torch.tensor(sp_sem).permute(2,0,1).float()
            # n = torch.tensor(n_sem_img).permute(2,0,1).float()
        return [q, k1, k2], index


class HabitatSemObjwiseEvalDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-2] for i in range(len(self.data_list))]))

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def pull_image(self, index):
        scene = self.data_list[index].split("/")[-2]
        scene_idx = self.scenes.index(scene)
        x_sem = cv2.imread(self.data_list[index].replace("rgb", "sem"))[...,0:1]
        semantic_img = Image.new(
            "P", (x_sem.shape[1], x_sem.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((x_sem.flatten() % 40).astype(np.uint8))
        semantic_img = np.array(semantic_img.convert("RGBA"))[...,:3]/255.
        q = torch.tensor(semantic_img).permute(2,0,1).float()
        return q, index, scene_idx

class AI2thorImageDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def augment(self, img):
        data_patches = np.stack([img[:, i * 32:(i + 1) * 32] for i in range(8)])
        index_list = np.arange(0, 8).tolist()
        random_cut = np.random.randint(1, 8)
        index_list = index_list[random_cut:] + index_list[:random_cut]
        permuted_patches = data_patches[index_list]
        augmented_img = np.concatenate(np.split(permuted_patches, 8, axis=0), 2)[0]
        return augmented_img

    def pull_image(self, index):
        x = plt.imread(self.data_list[index])
        l_x, l_y, _, _ = self.data_list[index].split("/")[-1].split(".png")[0].split("|")
        l_x, l_y = float(l_x), float(l_y)
        l_r = np.random.randint(8) * 45
        p = np.random.randint(9)
        if p == 0:
            x_aug_data = os.path.join("/".join(self.data_list[index].split("/")[:-1]), "|".join(['%.2f' % (l_x - 0.25), '%.2f' % (l_y-0.25), '%d' % l_r]) + "|0.png")
        elif p == 1:
            x_aug_data = os.path.join("/".join(self.data_list[index].split("/")[:-1]), "|".join(['%.2f' % (l_x - 0.25), '%.2f' % l_y, '%d' % l_r]) + "|0.png")
        elif p == 2:
            x_aug_data = os.path.join("/".join(self.data_list[index].split("/")[:-1]), "|".join(['%.2f' % (l_x - 0.25), '%.2f' % (l_y+0.25), '%d' % l_r]) + "|0.png")
        elif p == 3:
            x_aug_data = os.path.join("/".join(self.data_list[index].split("/")[:-1]), "|".join(['%.2f' % (l_x), '%.2f' % (l_y-0.25), '%d' % l_r]) + "|0.png")
        elif p == 4:
            x_aug_data = os.path.join("/".join(self.data_list[index].split("/")[:-1]), "|".join(['%.2f' % (l_x), '%.2f' % (l_y), '%d' % l_r]) + "|0.png")
        elif p == 5:
            x_aug_data = os.path.join("/".join(self.data_list[index].split("/")[:-1]), "|".join(['%.2f' % (l_x), '%.2f' % (l_y+0.25), '%d' % l_r]) + "|0.png")
        elif p == 6:
            x_aug_data = os.path.join("/".join(self.data_list[index].split("/")[:-1]), "|".join(['%.2f' % (l_x+0.25), '%.2f' % (l_y-0.25), '%d' % l_r]) + "|0.png")
        elif p == 7:
            x_aug_data = os.path.join("/".join(self.data_list[index].split("/")[:-1]), "|".join(['%.2f' % (l_x+0.25), '%.2f' % (l_y), '%d' % l_r]) + "|0.png")
        elif p == 8:
            x_aug_data = os.path.join("/".join(self.data_list[index].split("/")[:-1]), "|".join(['%.2f' % (l_x+0.25), '%.2f' % (l_y+0.25), '%d' % l_r]) + "|0.png")

        if os.path.exists(x_aug_data):
            x_aug = plt.imread(x_aug_data)
        else:
            x_aug = self.augment(x)
        q = torch.tensor(x[...,:3]).permute(2,0,1)
        k = torch.tensor(x_aug[...,:3]).permute(2,0,1)
        if self.noisydepth:
            q = torch.cat([q, torch.tensor(x[...,-1:]).permute(2,0,1)], 0)
            k = torch.cat([k, torch.tensor(x_aug[...,-1:]).permute(2,0,1)], 0)
        return [q, k], index

class AI2thorImageEvalDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def pull_image(self, index):
        # scene = self.data_list[index].split("/")[-2]
        x = plt.imread(self.data_list[index])
        q = torch.tensor(x[...,:3]).permute(2,0,1)
        if self.noisydepth:
            q = torch.cat([q, torch.tensor(x[...,-1:]).permute(2,0,1)], 0)
        return q, index


class HabitatObjectEvalDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth
        self.dataset = "mp3d"
        if "mp3d" in self.data_list[0]:
            self.dataset = "mp3d"
        elif "gibson" in self.data_list[0]:
            self.dataset = "gibson"
        elif "hm3d" in self.data_list[0]:
            self.dataset = "hm3d"

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def pull_image(self, index):
        x = plt.imread(self.data_list[index])[...,:3]

        if self.base_transform is not None:
            q = self.base_transform(Image.fromarray((x*255.).astype(np.uint8)))
        else:
            q = torch.tensor(x).permute(2,0,1)

        q_loc = joblib.load(self.data_list[index].replace('.png', '.dat.gz'))
        q_loc = torch.tensor([0] + list(q_loc['bbox']))
        return q, q_loc, index

    def draw_bbox(self, rgb: np.ndarray, bboxes: np.ndarray, bbox_categories) -> np.ndarray:
        imgHeight, imgWidth, _ = rgb.shape
        if bboxes.max() <= 1: bboxes[:, [0, 2]] *= imgWidth; bboxes[:, [1, 3]] *= imgHeight
        for i, bbox in enumerate(bboxes):
            rgb = cv2.rectangle(rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), int(5e-2 * imgHeight))
            label = CATEGORIES[self.dataset][bbox_categories[i]]
            rgb = cv2.putText(rgb, label, (int(bbox[0]), int(bbox[1]) + int(imgHeight * 0.1)), 0, 5e-3 * imgHeight, (0, 255, 255), int(2e-2 * imgHeight))
        return rgb


class HabitatObjectDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth
        self.dataset = "mp3d"
        if "mp3d" in self.data_list[0]:
            self.dataset = "mp3d"
        elif "gibson" in self.data_list[0]:
            self.dataset = "gibson"
        elif "hm3d" in self.data_list[0]:
            self.dataset = "hm3d"

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def add_bbox_noise(self, input_object_t, noise_amount=20, input_height=0.5, input_width=0.5):
        input_object_noised_t = input_object_t.copy()
        input_object_noised_t[:, 0] = np.maximum(input_object_noised_t[:, 0] + np.clip(np.random.standard_normal(input_object_t.shape[0]) * (noise_amount * input_width / 100.), -noise_amount, noise_amount), 0)
        input_object_noised_t[:, 1] = np.maximum(input_object_noised_t[:, 1] + np.clip(np.random.standard_normal(input_object_t.shape[0]) * (noise_amount * input_height / 100.), -noise_amount, noise_amount), 0)
        input_object_noised_t[:, 2] = np.minimum(input_object_noised_t[:, 2] + np.clip(np.random.standard_normal(input_object_t.shape[0]) * (noise_amount * input_width / 100.), -noise_amount, noise_amount), 1)
        input_object_noised_t[:, 3] = np.minimum(input_object_noised_t[:, 3] + np.clip(np.random.standard_normal(input_object_t.shape[0]) * (noise_amount * input_height / 100.), -noise_amount, noise_amount), 1)
        return input_object_noised_t

    def reduce_half(self, input_object_t):
        random_int = np.random.randint(10)
        if random_int == 0:
            random_int = np.random.randint(4)
            box_size = np.maximum(np.random.rand(), 0.5)
            input_object_t = input_object_t.copy()
            if random_int == 0:
                input_object_t[:, 3] = (input_object_t[:, 3] - input_object_t[:, 1]) * box_size + input_object_t[:, 1]
            elif random_int == 1:
                input_object_t[:, 2] = (input_object_t[:, 2] - input_object_t[:, 0]) * box_size + input_object_t[:, 0]
            elif random_int == 2:
                input_object_t[:, 0] = (input_object_t[:, 2] - input_object_t[:, 0]) * box_size + input_object_t[:, 0]
            elif random_int == 3:
                input_object_t[:, 1] = (input_object_t[:, 3] - input_object_t[:, 1]) * box_size + input_object_t[:, 1]
        return input_object_t

    def pull_image(self, index):
        x = plt.imread(self.data_list[index])[...,:3]
        same_obj_images = list(np.sort(glob.glob(os.path.join("/".join(self.data_list[index].split("/")[:-1]), '*.png'))))
        same_obj_images.remove(self.data_list[index])

        q_loc = joblib.load(self.data_list[index].replace('.png', '.dat.gz'))
        q_bbox = np.array(q_loc['bbox']).reshape(-1, 4)
        # rgb_test_a = self.draw_bbox(x.copy(), q_bbox.copy(), [q_loc['bbox_category'].copy()])
        q_bbox = torch.tensor([0] + list(q_bbox[0]))

        if np.random.randint(3) == 0 or len(same_obj_images) == 0:
            x_aug = x.copy()
            k_loc = q_loc.copy()
            k_bbox = np.array(k_loc['bbox']).reshape(-1, 4)
            k_bbox = self.reduce_half(k_bbox)
            input_width = (k_bbox[:, 2] - k_bbox[:, 0])
            input_height = (k_bbox[:, 3] - k_bbox[:, 1])
            k_bbox = self.add_bbox_noise(k_bbox, noise_amount=20, input_height=input_height, input_width=input_width)
            # rgb_test_b = self.draw_bbox(x_aug.copy(), k_bbox.copy(), [k_loc['bbox_category'].copy()])
            k_bbox = torch.tensor([0] + list(k_bbox[0]))
        else:
            # same_obj_images = np.stack(same_obj_images)
            idx = np.random.randint(len(same_obj_images))
            x_aug = plt.imread(same_obj_images[idx])[...,:3]
            k_loc = joblib.load(same_obj_images[idx].replace(".png",".dat.gz"))
            k_bbox = np.array(k_loc['bbox']).reshape(-1, 4)
            # rgb_test_b = self.draw_bbox(x_aug.copy(), k_bbox.copy(), [k_loc['bbox_category'].copy()])
            k_bbox = self.reduce_half(k_bbox)
            k_bbox = torch.tensor([0] + list(k_bbox[0]))


        if self.base_transform is not None:
            q = self.base_transform(Image.fromarray((x*255.).astype(np.uint8)))
            k = self.base_transform(Image.fromarray((x_aug*255.).astype(np.uint8)))
        else:
            q = torch.tensor(x).permute(2,0,1)
            k = torch.tensor(x_aug).permute(2,0,1)
        return [q, k], [q_bbox, k_bbox], index

    def draw_bbox(self, rgb: np.ndarray, bboxes: np.ndarray, bbox_categories) -> np.ndarray:
        imgHeight, imgWidth, _ = rgb.shape
        if bboxes.max() <= 1: bboxes[:, [0, 2]] *= imgWidth; bboxes[:, [1, 3]] *= imgHeight
        for i, bbox in enumerate(bboxes):
            rgb = cv2.rectangle(rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), int(5e-2 * imgHeight))
            label = CATEGORIES[self.dataset][bbox_categories[i]]
            rgb = cv2.putText(rgb, label, (int(bbox[0]), int(bbox[1]) + int(imgHeight * 0.1)), 0, 5e-3 * imgHeight, (0, 255, 255), int(2e-2 * imgHeight))
        return rgb


class AI2ThorObjectDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def add_bbox_noise(self, input_object_t, noise_amount=20, input_height=0.5, input_width=0.5):
        input_object_noised_t = input_object_t.copy()
        input_object_noised_t[:, 0] = np.maximum(input_object_noised_t[:, 0] + np.clip(np.random.standard_normal(input_object_t.shape[0]) * (noise_amount * input_width / 100.), -noise_amount, noise_amount), 0)
        input_object_noised_t[:, 1] = np.maximum(input_object_noised_t[:, 1] + np.clip(np.random.standard_normal(input_object_t.shape[0]) * (noise_amount * input_height / 100.), -noise_amount, noise_amount), 0)
        input_object_noised_t[:, 2] = np.minimum(input_object_noised_t[:, 2] + np.clip(np.random.standard_normal(input_object_t.shape[0]) * (noise_amount * input_width / 100.), -noise_amount, noise_amount), 1)
        input_object_noised_t[:, 3] = np.minimum(input_object_noised_t[:, 3] + np.clip(np.random.standard_normal(input_object_t.shape[0]) * (noise_amount * input_height / 100.), -noise_amount, noise_amount), 1)
        return input_object_noised_t

    def reduce_half(self, input_object_t):
        random_int = np.random.randint(10)
        if random_int == 0:
            random_int = np.random.randint(4)
            input_object_t = input_object_t.copy()
            if random_int == 0:
                input_object_t[:, 3] = (input_object_t[:, 3] - input_object_t[:, 1]) * 0.5 + input_object_t[:, 1]
            elif random_int == 1:
                input_object_t[:, 2] = (input_object_t[:, 2] - input_object_t[:, 0]) * 0.5 + input_object_t[:, 0]
            elif random_int == 2:
                input_object_t[:, 0] = (input_object_t[:, 2] - input_object_t[:, 0]) * 0.5 + input_object_t[:, 0]
            elif random_int == 3:
                input_object_t[:, 1] = (input_object_t[:, 3] - input_object_t[:, 1]) * 0.5 + input_object_t[:, 1]
        return input_object_t

    def pull_image(self, index):
        img_path = "|".join(self.data_list[index].split("|")[:-7]).replace("objects", "image") + ".png"
        obj_cat = self.data_list[index].split("|")[5]
        obj_loc = np.stack([float(j) for j in np.stack(self.data_list[index].split("|")[6:9])])
        x = plt.imread(img_path)
        q = torch.tensor(x[...,:3]).permute(2,0,1)

        q_loc = joblib.load(self.data_list[index])
        q_bbox = np.array(q_loc['bboxes']).reshape(-1, 4)
        q_loc = torch.tensor([0] + list(q_bbox[0]))

        same_scene_obj = glob.glob(os.path.join("/".join(self.data_list[index].split("/")[:-1]), '*.dat.gz'))
        same_scene_obj_loc = np.stack([[float(j) for j in np.stack(i.split("|")[6:9])] for i in same_scene_obj])
        dist = ((same_scene_obj_loc - obj_loc[None])**2).sum(-1)**0.5
        category = np.array([i.split("|")[5] for i in same_scene_obj])
        same_objs = np.where((dist == 0) & (category == obj_cat))[0]
        randi = random.randint(0, len(same_objs)-1)
        same_idx = same_objs[randi]
        k_loc = joblib.load(same_scene_obj[same_idx])
        k_bbox = np.array(k_loc['bboxes']).reshape(-1, 4)
        object_path = same_scene_obj[same_idx]
        same_img_path = "|".join(object_path.split("|")[:-7]).replace("objects", "image") + ".png"
        x_aug = plt.imread(same_img_path)
        k = torch.tensor(x_aug[...,:3]).permute(2,0,1)
        k_bbox = self.reduce_half(k_bbox)
        k_bbox = self.add_bbox_noise(k_bbox, noise_amount=0.01, input_height=(k_bbox[:, 3] - k_bbox[:, 1]), input_width=(k_bbox[:, 2] - k_bbox[:, 0]))
        k_loc = torch.tensor([0] + list(k_bbox[0]))
        return [q, k], [q_loc, k_loc], index


class AI2ThorObjectEvalDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def pull_image(self, index):
        img_path = "|".join(self.data_list[index].split("|")[:-7]).replace("objects", "image") + ".png"
        # obj_cat = self.data_list[index].split("|")[5]
        # obj_loc = np.stack([float(j) for j in np.stack(self.data_list[index].split("|")[6:9])])
        x = plt.imread(img_path)
        q = torch.tensor(x[...,:3]).permute(2,0,1)

        q_loc = joblib.load(self.data_list[index])
        q_bbox = np.array(q_loc['bboxes']).reshape(-1, 4)
        q_loc = torch.tensor([0] + list(q_bbox[0]))

        # x = plt.imread(self.data_list[index])
        # q = torch.tensor(x[...,:3]).permute(2,0,1)
        # q_loc = joblib.load(self.data_list[index].replace('.png', '.dat.gz').replace("image", "bboxes"))
        # bbox_idx = np.random.randint(len(q_loc))
        # q_bbox = np.array(q_loc['bboxes'][bbox_idx]).reshape(-1, 4)
        # q_loc = torch.tensor([0] + list(q_bbox[0]))
        return q, q_loc, index


class HabitatRGBObjDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-3] for i in range(len(self.data_list))]))
        self.max_object = 10
        hists = joblib.load("/".join(data_list[0].split("/")[:5]) + '/object_hists.dat.gz')
        self.hists = hists['hists']
        for i, hist_path in enumerate( hists['data_path']):
            hists['data_path'][i] = os.path.join("/".join(data_list[0].split("/")[:-5]), "/".join(hist_path.split("/")[-5:]))
        self.hists_datapath = hists['data_path']
        self.num_camera = 12
        self.unique_places = ['']
        self.place_mapping = {
            'living room': ['living room'],
            # 'familyroom_lounge': ['living room', 'familyroom_lounge', 'lounge', 'tv'],
            # 'lounge': ['living room', 'familyroom_lounge', 'lounge', 'tv'],
            'tv': ['tv'],
            'bedroom': ['bedroom'],
            'kitchen': ['kitchen'],
            'closet': ['closet'],
            'dining room': ['dining room', 'dining booth'],
            'dining booth': ['dining room', 'dining booth'],
            'bathroom': ['bathroom'],
            'toilet': ['toilet'],
            'hallway': ['hallway'],
            # 'office': ['office', 'classroom', 'meetingroom_conferenceroom', 'library'],
            # 'classroom': ['office', 'classroom', 'meetingroom_conferenceroom', 'library'],
            # 'meetingroom_conferenceroom': ['office', 'classroom', 'meetingroom_conferenceroom', 'library'],
            # 'library': ['office', 'classroom', 'meetingroom_conferenceroom', 'library'],
            }

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def augment(self, rgb, obj):
        patch_size = rgb.shape[1]//self.num_camera
        data_patches = np.stack([rgb[:, i * patch_size:(i + 1) * patch_size] for i in range(self.num_camera)])
        index_list = np.arange(0, self.num_camera).tolist()
        random_cut = np.random.randint(self.num_camera)
        index_list = index_list[random_cut:] + index_list[:random_cut]
        permuted_patches = data_patches[index_list]
        rgb_roted = np.concatenate(np.split(permuted_patches, self.num_camera, axis=0), 2)[0]
        a = float(random_cut * 21)
        width = float(rgb.shape[1])
        bboxes = obj['bboxes'].copy()
        for bbox in bboxes:
            if bbox[0] < a/width:
                bbox[0] += (width - a)/width
                bbox[2] += (width - a)/width
            else:
                bbox[0] -= a/width
                bbox[2] -= a/width
        obj['bboxes'] = bboxes
        # bboxes = obj['bboxes']
        # bbox_category = obj['bbox_categories']
        # for i, bbox in enumerate(bboxes):
        #     label = CATEGORIES['mp3d'][bbox_category[i]]
        #     imgHeight, imgWidth, _ = rgb_roted.shape
        #     cv2.rectangle(rgb_roted, (int(bbox[0]*imgWidth), int(bbox[1]*imgHeight)), (int(bbox[2]*imgWidth), int(bbox[3]*imgHeight)), [255,255,0], 1)
        #     if len(bbox_category) > 0:
        #         cv2.putText(rgb_roted, label, (int(bbox[0]*imgWidth), int(bbox[1]*imgHeight) + 10), 0, 5e-3 * imgHeight, (183, 115, 48), 1)
        return rgb_roted, obj

    def pull_image(self, index):
        x = plt.imread(self.data_list[index])[...,:3]
        x_obj = joblib.load(self.data_list[index].replace('_rgb.png', '.dat.gz').replace('image', 'object'))
        x_obj_out = np.zeros((self.max_object, 4))
        x_obj_out[:, 2:] = 1.
        x_obj_category_out = np.zeros((self.max_object))
        x_obj_out[:len(x_obj['bboxes'])] = x_obj['bboxes'][:self.max_object]
        x_obj_category_out[:len(x_obj['bbox_categories'])] = x_obj['bbox_categories'][:self.max_object]

        place = self.data_list[index].split("/")[-2]
        match_places = self.place_mapping[place]
        same_place_list = [self.data_list[i] for i in range(len(self.data_list)) if self.data_list[i].split("/")[-2] in match_places]
        if len(same_place_list) > 1:
            same_place_list.remove(self.data_list[index])
        idx = np.random.randint(len(same_place_list))
        sp_sample = same_place_list[idx]
        # data_ii = self.hists_datapath.index(self.data_list[index])
        # hist = self.hists[data_ii]
        # similarity = -np.sum(hist[None] * np.log(hist[None] / (self.hists + 0.0001)),-1)
        # cands = np.argsort(-similarity)[:100]
        # cands = cands[abs(cands - data_ii)>10000]
        # if len(cands) == 0:
        #     cands = np.argsort(-similarity)[:100]
        # idx = cands[random.choices(np.arange(len(cands)))[0]]
        # sp_sample = self.hists_datapath[idx]
        sp = plt.imread(sp_sample)[...,:3]
        sp_obj = joblib.load(sp_sample.replace('_rgb.png', '.dat.gz'))
        sp_obj_out = np.zeros((self.max_object, 4))
        sp_obj_out[:, 2:] = 1.
        sp_obj_category_out = np.zeros((self.max_object))
        sp_obj_out[:len(sp_obj['bboxes'])] = sp_obj['bboxes'][:self.max_object]
        sp_obj_category_out[:len(sp_obj['bbox_categories'])] = sp_obj['bbox_categories'][:self.max_object]

        x_aug, x_aug_obj = self.augment(sp, sp_obj)
        x_aug_obj_out = np.zeros((self.max_object, 4))
        x_aug_obj_out[:, 2:] = 1.
        x_aug_obj_category_out = np.zeros((self.max_object))
        x_aug_obj_out[:len(x_aug_obj['bboxes'])] = x_aug_obj['bboxes'][:self.max_object]
        x_aug_obj_category_out[:len(x_aug_obj['bbox_categories'])] = x_aug_obj['bbox_categories'][:self.max_object]

        if self.base_transform is not None:
            q = self.base_transform(Image.fromarray((x*255.).astype(np.uint8)))
            k1 = self.base_transform(Image.fromarray((x_aug*255.).astype(np.uint8)))
            k2 = self.base_transform(Image.fromarray((sp*255.).astype(np.uint8)))
        else:
            q = torch.tensor(x).permute(2,0,1).float()
            k1 = torch.tensor(x_aug).permute(2,0,1).float()
            k2 = torch.tensor(sp).permute(2,0,1).float()
        # query, same_place_rot, same_place
        return [q, k1, k2], [x_obj_out, x_aug_obj_out, sp_obj_out], [x_obj_category_out, x_aug_obj_category_out, sp_obj_category_out], index


class HabitatRGBObjEvalDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-2] for i in range(len(self.data_list))]))
        self.max_object = 10

    def __getitem__(self, index):
        return self.pull_image(index)

    def __len__(self):
        return len(self.data_list)

    def pull_image(self, index):
        x = plt.imread(self.data_list[index])[...,:3]
        scene = self.data_list[index].split("/")[-2]
        scene_idx = self.scenes.index(scene)
        x_obj = joblib.load(self.data_list[index].replace('_rgb.png', '.dat.gz').replace('image', 'object'))

        x_obj_out = np.zeros((self.max_object, 4))
        x_obj_out[:, 2:] = 1.
        x_obj_category_out = np.zeros((self.max_object))
        x_obj_out[:len(x_obj['bboxes'])] = x_obj['bboxes'][:self.max_object]
        x_obj_category_out[:len(x_obj['bbox_categories'])] = x_obj['bbox_categories'][:self.max_object]

        if self.base_transform is not None:
            q = self.base_transform(Image.fromarray((x*255.).astype(np.uint8)))
        else:
            q = torch.tensor(x).permute(2,0,1).float()
        return q, x_obj_out, x_obj_category_out, scene_idx
