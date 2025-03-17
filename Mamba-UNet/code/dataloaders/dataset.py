import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image


class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            # # 计算要选择的数据数量（10%）
            # num_samples = len(self.sample_list)
            # num_samples_to_select = int(1 * num_samples)
            # # 随机选择数据
            # selected_samples = random.sample(self.sample_list, num_samples_to_select)
            # self.sample_list = selected_samples

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image_path = os.path.join(self._base_dir, "data/images", f"{case}.png")
        label_path = os.path.join(self._base_dir, "data/gt", f"{case}.png")

        # 读取彩色图像，不需要转换
        image = cv2.imread(image_path)  # 直接读取BGR格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # 标签仍然是灰度图

        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 调整维度顺序：从[H,W,3]变为[3,H,W]
        image = np.transpose(image, (2, 0, 1))
        
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

    def preprocess_oct(self, image):
        """OCT图像特定的预处理"""
        # 对比度增强
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # image = clahe.apply(image)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        return image

class BaseDataSets_Synapse(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        image_path = os.path.join(self._base_dir, "train_npz", f"{case}.png")
        label_path = os.path.join(self._base_dir, "train_npz", f"{case}_label.png")  # 假设标签图像以"_label"结尾

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # 读取标签图像

        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)  # 旋转RGB图像
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)  # 旋转RGB图像
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # 确保图像格式正确
        if isinstance(image, np.ndarray):
            if image.shape[0] == 3:  # 如果是 [C,H,W] 格式
                image = np.transpose(image, (1, 2, 0))  # 转换为 [H,W,C]
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3 and image.shape[0] == 3:  # 如果是 [C,H,W] 格式
                image = image.permute(1, 2, 0).contiguous().numpy()  # 转换为 [H,W,C]
            else:
                image = image.numpy()
                if len(image.shape) == 3 and image.shape[0] == 1:  # 如果是 [1,H,W] 格式
                    image = image.squeeze(0)  # 转换为 [H,W]

        # 确保值范围在 [0,1] 之间
        image = (image * 255).astype(np.uint8)
        label = label.astype(np.uint8)

        # apply augmentations
        if len(image.shape) == 2:  # 如果是灰度图
            image = Image.fromarray(image, mode='L')
        else:  # 如果是RGB图
            image = Image.fromarray(image, mode='RGB')
        
        label = Image.fromarray(label, mode='L')

        image_weak = augmentations.cta_apply(image, ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(label, ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image": to_tensor(image),
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
            "label": label_aug
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        if len(image.shape) == 3:
            # 对于 RGB 图像 (H,W,C)
            x, y, c = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        else:
            # 对于灰度图像 (H,W)
            x, y = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        
        # 注意image现在是[3,H,W]格式
        if random.random() > 0.5:
            image = np.transpose(image, (1, 2, 0))  # [3,H,W] -> [H,W,3]
            image, label = random_rot_flip(image, label)
            image = np.transpose(image, (2, 0, 1))  # [H,W,3] -> [3,H,W]
        elif random.random() > 0.5:
            image = np.transpose(image, (1, 2, 0))  # [3,H,W] -> [H,W,3]
            image, label = random_rotate(image, label)
            image = np.transpose(image, (2, 0, 1))  # [H,W,3] -> [3,H,W]
            
        x, y = image.shape[1:]  # 获取高和宽
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)  # 保持通道数不变
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class RandomGenerator_w(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample

class RandomGenerator_s(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = color_jitter(image).type("torch.FloatTensor")
        image = rand_affine(image).type("torch.FloatTensor")
        image = gaussian_blur(image).type("torch.FloatTensor")
        image = rand_gray(image).type("torch.FloatTensor")
#         grid_mask = Grid_Mask()
#         image = grid_mask(image).type("torch.FloatTensor")
#         image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        
        sample = {"image": image, "label": label}
        return sample
    