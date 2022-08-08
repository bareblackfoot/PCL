from PIL import ImageFilter
import random
import torchvision.datasets as datasets
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torch
import joblib, glob, os, cv2
from habitat_sim.utils.common import d3_40_colors_rgb
from PIL import Image


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
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-2] for i in range(len(self.data_list))]))

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
        scene = self.data_list[index].split("/")[-2]
        scene_idx = self.scenes.index(scene)
        scene_data_list = [self.data_list[i] for i in range(len(self.data_list)) if self.data_list[i].split("/")[-2] == scene]
        scene_data_list.remove(self.data_list[index])
        idx = np.random.randint(len(scene_data_list))
        negative_sample = scene_data_list[idx]
        n = plt.imread(negative_sample)
        n = torch.tensor(n[...,:3]).permute(2,0,1)

        x = plt.imread(self.data_list[index])
        x_aug = self.augment(x)
        q = torch.tensor(x[...,:3]).permute(2,0,1)
        k = torch.tensor(x_aug[...,:3]).permute(2,0,1)
        if self.noisydepth:
            q = torch.cat([q, torch.tensor(x[...,-1:]).permute(2,0,1)], 0)
            k = torch.cat([k, torch.tensor(x_aug[...,-1:]).permute(2,0,1)], 0)
        return [q, k, n], index, scene_idx

class HabitatImageEvalDataset(data.Dataset):
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
        q = torch.tensor(x[...,:3]).permute(2,0,1)
        if self.noisydepth:
            q = torch.cat([q, torch.tensor(x[...,-1:]).permute(2,0,1)], 0)
        return q, index, scene_idx



class HabitatImageSemDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-3] for i in range(len(self.data_list))]))

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
        scene_data_list = [self.data_list[i] for i in range(len(self.data_list)) if self.data_list[i].split("/")[-3] == scene]
        scene_data_list.remove(self.data_list[index])
        idx = np.random.randint(len(scene_data_list))
        negative_sample = scene_data_list[idx]
        n = plt.imread(negative_sample)
        n_sem = cv2.imread(negative_sample.replace("rgb", "sem"))[...,0:1]
        semantic_img = Image.new(
            "P", (n_sem.shape[1], n_sem.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((n_sem.flatten() % 40).astype(np.uint8))
        semantic_img = np.array(semantic_img.convert("RGBA"))[...,:3]/255.
        n = torch.tensor(np.concatenate([n[...,:3], semantic_img], -1)).permute(2,0,1).float()

        x = plt.imread(self.data_list[index])
        x_sem = cv2.imread(self.data_list[index].replace("rgb", "sem"))[...,0:1]
        semantic_img = Image.new(
            "P", (x_sem.shape[1], x_sem.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((x_sem.flatten() % 40).astype(np.uint8))
        semantic_img = np.array(semantic_img.convert("RGBA"))[...,:3]/255.
        x_aug, x_aug_sem = self.augment(x, semantic_img)
        q = torch.tensor(np.concatenate([x[...,:3], semantic_img], -1)).permute(2,0,1).float()
        k = torch.tensor(np.concatenate([x_aug[...,:3], x_aug_sem], -1)).permute(2,0,1).float()
        return [q, k, n], index, scene_idx


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
        q = torch.tensor(np.concatenate([x[...,:3], semantic_img], -1)).permute(2,0,1).float()
        return q, index, scene_idx


class HabitatSemDataset(data.Dataset):
    def __init__(self, data_list, base_transform=None, noisydepth=False):
        self.data_list = data_list
        self.base_transform = base_transform
        self.noisydepth = noisydepth
        self.scenes = sorted(np.unique([self.data_list[i].split("/")[-3] for i in range(len(self.data_list))]))

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
        scene = self.data_list[index].split("/")[-3]
        scene_idx = self.scenes.index(scene)
        scene_data_list = [self.data_list[i] for i in range(len(self.data_list)) if self.data_list[i].split("/")[-3] == scene]
        scene_data_list.remove(self.data_list[index])
        idx = np.random.randint(len(scene_data_list))
        negative_sample = scene_data_list[idx]
        n_sem = cv2.imread(negative_sample.replace("rgb", "sem"))[...,0:1]
        semantic_img = Image.new(
            "P", (n_sem.shape[1], n_sem.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((n_sem.flatten() % 40).astype(np.uint8))
        semantic_img = np.array(semantic_img.convert("RGBA"))[...,:3]/255.
        n = torch.tensor(semantic_img).permute(2,0,1).float()

        x_sem = cv2.imread(self.data_list[index].replace("rgb", "sem"))[...,0:1]
        semantic_img = Image.new(
            "P", (x_sem.shape[1], x_sem.shape[0])
        )
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((x_sem.flatten() % 40).astype(np.uint8))
        semantic_img = np.array(semantic_img.convert("RGBA"))[...,:3]/255.
        x_aug_sem = self.augment(semantic_img)
        q = torch.tensor(semantic_img).permute(2,0,1).float()
        k = torch.tensor(x_aug_sem).permute(2,0,1).float()
        return [q, k, n], index, scene_idx


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


class HabitatObjectDataset(data.Dataset):
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
        x = plt.imread(self.data_list[index])
        same_obj_images = glob.glob(os.path.join("/".join(self.data_list[index].split("/")[:-1]), '*.png'))
        same_obj_data = glob.glob(os.path.join("/".join(self.data_list[index].split("/")[:-1]), '*.dat.gz'))
        idx = np.random.randint(len(same_obj_images))
        x_aug = plt.imread(same_obj_images[idx])
        q = torch.tensor(x[...,:3]).permute(2,0,1)
        k = torch.tensor(x_aug[...,:3]).permute(2,0,1)

        q_loc = joblib.load(self.data_list[index].replace('.png', '.dat.gz'))

        q_bbox = np.array(q_loc['bboxes']).reshape(-1, 4)
        # input_width = (q_bbox[:, 2] - q_bbox[:, 0])
        # input_height = (q_bbox[:, 3] - q_bbox[:, 1])
        # q_bbox = self.add_bbox_noise(q_bbox, noise_amount=20, input_height=input_height, input_width=input_width)
        q_loc = torch.tensor([0] + list(q_bbox[0]))
        k_loc = joblib.load(same_obj_data[idx])
        k_bbox = np.array(k_loc['bboxes']).reshape(-1, 4)
        k_bbox = self.reduce_half(k_bbox)
        input_width = (k_bbox[:, 2] - k_bbox[:, 0])
        input_height = (k_bbox[:, 3] - k_bbox[:, 1])

        k_bbox = self.add_bbox_noise(k_bbox, noise_amount=20, input_height=input_height, input_width=input_width)
        k_loc = torch.tensor([0] + list(k_bbox[0]))
        return [q, k], [q_loc, k_loc], index


class HabitatObjectEvalDataset(data.Dataset):
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
        q = torch.tensor(x[...,:3]).permute(2,0,1)

        q_loc = joblib.load(self.data_list[index].replace('.png', '.dat.gz'))
        q_loc = torch.tensor([0] + list(q_loc['bboxes']))
        return q, q_loc, index
