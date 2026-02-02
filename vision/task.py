import random
from abc import ABC, abstractmethod
import numpy as np
import os
import requests
from io import BytesIO
import zipfile
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = "./data"

def get_transform(size: int, statistics):
    mean, std = statistics
    transform = []

    transform.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transforms.Compose(transform)

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __getitem__(self, idx):
        data, label = self.base_dataset[idx]

        if hasattr(self.base_dataset, 'indices'):
            original_idx = self.base_dataset.indices[idx]
        else:
            original_idx = idx

        if hasattr(self.base_dataset, 'chunk_indices'):
            chunk_idx = self.base_dataset.chunk_indices[idx]
        else:
            chunk_idx = -1
        return data, label, original_idx, chunk_idx  # <-- include index

    def __len__(self):
        return len(self.base_dataset)


class Task(ABC):
    _train_datasets = None
    _test_dataset = None

    def __init__(
            self,
            mode: str='sample',
            n_chunks:int=10,
            make_test_loader: bool = True,
            access: str = "limited",
            test_access: str = "same",
            chunk_size: int = 0,
            test_chunk_size: int = 0,
            seed: int = 0,
            warm_start_subset_ratio: int = 10,
    ):
        self.mode = mode
        self.n_chunks = n_chunks
        self.level = 0
        self.seed = seed

        self._init_dataset(make_test_loader, access, test_access, chunk_size, test_chunk_size, warm_start_subset_ratio=warm_start_subset_ratio)

    @abstractmethod
    def _get_dataset(self):
        pass

    def set_level(self, level: int, batch_size: int = 64):
        if 0 <= level < self.n_chunks:
            self.level = level
        else:
            raise ValueError(f"Level should be in [0, {self.n_chunks})")

        dataset = self._train_datasets[self.level]
        # print(f'Dataset type at level {level}: {type(dataset)}')  # 데이터셋 타입을 출력

        return DataLoader(IndexedDataset(dataset), batch_size=batch_size, shuffle=True)

    def get_indexed_test_loader(self):
        return DataLoader(IndexedDataset(self._test_datasets[self.level]), batch_size=64, shuffle=False)

    def test(self, model: nn.Module, device: str, full=False, train=False, log_features=False, log_input_grad=False):
        model.eval()
        info = {}
        if full and train:
            raise ValueError("full and train cannot be True at the same time")
        if full:
            test_loader = DataLoader(self._test_dataset, batch_size=64, shuffle=False)
        elif train:
            test_loader = DataLoader(self._train_datasets[self.level], batch_size=64, shuffle=False)
        else:
            test_loader = DataLoader(self._test_datasets[self.level], batch_size=64, shuffle=False)

        if log_features:
            model.enable_hooks()
            sample_num = 1000
            num = 0
            total_features = []
        if log_input_grad:
            total_input_grads = []

        correct = 0
        total = 0
        for (images, labels) in test_loader:
            images, labels = images.to(device), labels.to(device)

            if log_input_grad:
                images.requires_grad_()

            # forward
            outputs = model(images)

            if log_input_grad:
                ce_loss = F.cross_entropy(outputs, labels)
                ce_loss.backward()
                total_input_grads += images.grad.flatten(start_dim=1).norm(dim=1).detach().cpu().numpy().tolist()


            # log feature covariance and activation pattern
            if log_features and num < sample_num:
                acts = model.get_activations()
                total_features.append(acts["backbone_output"].flatten(start_dim=1).detach().cpu())
                num += labels.size(0)

            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if log_features:
            model.disable_hooks()
            total_features = torch.cat(total_features, dim=0)[:sample_num, :]
            feature_covariance_matrix = (total_features @ total_features.T) / torch.sum(total_features ** 2) + 1e-7
            info['feature_covariance_matrix'] = feature_covariance_matrix
        if log_input_grad:
            info['avg_input_grad'] = sum(total_input_grads)/len(total_input_grads)
        return correct / total, info

    def get_test_loader(self):
        return DataLoader(self._test_dataset, batch_size=64, shuffle=False)

    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    def classes(self):
        return self._test_dataset.classes

    def _init_dataset(self, make_test_loader:bool, access: str, test_access: str, chunk_size: int, test_chunk_size: int, warm_start_subset_ratio: int = 10):
        train_dataset, test_dataset = self._get_dataset()

        n_train_data = len(train_dataset)
        n_test_data = len(test_dataset)

        self.total_n_train_data = n_train_data
        self.total_n_test_data = n_test_data

        np.random.seed(self.seed)
        train_indices = np.random.permutation(n_train_data)
        test_indices = np.random.permutation(n_test_data)
        self._train_dataset = train_dataset
        self._train_datasets = []
        self._test_dataset = test_dataset # test dataset is shared among all levels
        self._test_datasets = []
        if self.mode=="sample":
            if self.n_chunks == 2:  # warm start in Hare & Tortoise setting
                num_warm_start_data = n_train_data // (100 // warm_start_subset_ratio)
                print(f"Warm start training for {num_warm_start_data} samples, fine-tuning for {n_train_data} samples")
                self._train_datasets = [
                    Subset(train_dataset, train_indices[:num_warm_start_data]),
                    Subset(train_dataset, train_indices[:])
                ]
                self._unseen_dataset = Subset(train_dataset, train_indices[num_warm_start_data:])
                self._test_datasets = [test_dataset, test_dataset]
            else:
                for i in range(self.n_chunks):  # continual full and limited setting in Hare & Tortoise
                    train_ioi = get_chunk_idx(access, len(train_indices), self.n_chunks, chunk_size, i)
                    chunk_train_indices = train_indices[train_ioi]

                    test_ioi = get_chunk_idx(test_access, len(test_indices), self.n_chunks, test_chunk_size, i)
                    chunk_test_indices = test_indices[test_ioi]

                    train_subset = Subset(train_dataset, chunk_train_indices)

                    self._train_datasets.append(train_subset)
                    if make_test_loader:
                        test_subset = Subset(test_dataset, chunk_test_indices)
                        self._test_datasets.append(test_subset)

        elif self.mode=="class":  # class-incremental setting in Continual BackProp
            classes = np.random.permutation(range(len(train_dataset.classes)))
            n_class = len(classes) // self.n_chunks
            for i in range(self.n_chunks):
                start_idx = i * n_class if access == 'limited' else 0
                chunk_train_indices = np.where(np.isin(train_dataset.targets, classes[start_idx:(i + 1) * n_class]))[0]
                chunk_test_indices = np.where(np.isin(test_dataset.targets, classes[start_idx:(i + 1) * n_class]))[0]
                train_subset = Subset(train_dataset, chunk_train_indices)

                self._train_datasets.append(train_subset)
                if make_test_loader:
                    test_subset = Subset(test_dataset, chunk_test_indices)
                    self._test_datasets.append(test_subset)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from ['sample', 'class']")

        self.add_chunk_id()

    def add_chunk_id(self):
        used_indices = np.array([])
        chunk_indices = []
        for dataset in self._train_datasets:
            new_indices = np.setdiff1d(dataset.indices, used_indices)
            chunk_indices.append(new_indices)
            used_indices = np.concatenate((used_indices, dataset.indices))

        for dataset in self._train_datasets:
            set_ci = np.zeros_like(dataset.indices)
            for i, ci in enumerate(chunk_indices):
                mask = np.isin(dataset.indices, ci)
                set_ci[mask] = i
            dataset.chunk_indices = set_ci

def get_chunk_idx(access, dataset_len, n_chunks, chunk_size, i):
    is_chunk_size_zero = False
    if chunk_size == 0:
        is_chunk_size_zero = True
        chunk_size = dataset_len // n_chunks

    if access == 'limited':
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        indices = range(start_idx, end_idx)
    elif access == 'same':
        start_idx = 0
        if not is_chunk_size_zero:
            end_idx = min(dataset_len, chunk_size)
        else:
            end_idx = dataset_len
        indices = range(start_idx, end_idx)
    elif access == 'full':
        start_idx = 0
        end_idx = (i + 1) * chunk_size
        indices = range(start_idx, end_idx)
    elif access == 'stream':
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size

        start_idx %= dataset_len
        end_idx %= dataset_len

        if start_idx < end_idx:
            indices = range(start_idx, end_idx)
        else:
            indices = list(range(start_idx, dataset_len)) + list(range(0, end_idx))
    elif access == 'sample':
        indices = random.choices(list(range(dataset_len)), k=chunk_size)
    return indices


class CIFAR10(Task):
    def _get_dataset(self):
        train_mean, train_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        test_mean, test_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        size = 32
        train_transform = get_transform(
            size=size,
            statistics=(train_mean, train_std)
        )
        test_transform = get_transform(
            size=size,
            statistics=(test_mean, test_std)
        )

        train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

        self.train_mean, self.train_std = train_mean, train_std
        self.test_mean, self.test_std = test_mean, test_std
        return train_dataset, test_dataset


    @property
    def shape(self):
        return [3, 32, 32]


class CIFAR100(Task):
    def _get_dataset(self):
        train_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        train_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        test_mean = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
        test_std = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)
        size = 32
        train_transform = get_transform(
            size=size,
            statistics=(train_mean, train_std)
        )
        test_transform = get_transform(
            size=size,
            statistics=(test_mean, test_std)
        )
        # Train dataset
        train_dataset = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=test_transform)

        self.train_mean, self.train_std = train_mean, train_std
        self.test_mean, self.test_std = test_mean, test_std
        return train_dataset, test_dataset

    @property
    def shape(self):
        return [3, 32, 32]



def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class TinyImageNet(Task):
    def _get_dataset(self):
        root_dir = DATA_DIR
        train_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'train')
        val_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'val')

        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            # Download Tiny ImageNet dataset
            response = requests.get("http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            if response.status_code == 200:
                with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(root_dir)
            else:
                raise Exception(f"Failed to download dataset, status code: {response.status_code}")

            # Create train and validation directories
            create_dir(train_dir)
            create_dir(val_dir)

            # Move train data to train_dir
            # os.rename(os.path.join(root_dir, 'tiny-imagenet-200', 'train'), train_dir)

            # Separate validation images into separate sub-folders
            self._organize_val_dir(root_dir, val_dir)


        train_mean, train_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        test_mean, test_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        size = 64
        train_transform = get_transform(
            size=size,
            statistics=(train_mean, train_std)
        )
        test_transform = get_transform(
            size=size,
            statistics=(test_mean, test_std)
        )

        # Load datasets
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=val_dir, transform=test_transform)

        self.train_mean, self.train_std = train_mean, train_std
        self.test_mean, self.test_std = test_mean, test_std
        return train_dataset, test_dataset

    def _organize_val_dir(self, root_dir, val_dir):
        # Organize validation directory
        val_annotations_file = os.path.join(root_dir, 'tiny-imagenet-200', 'val', 'val_annotations.txt')
        val_images_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'val', 'images')

        # Read validation annotations
        with open(val_annotations_file, 'r') as f:
            data = f.readlines()

        val_img_dict = {}
        for line in data:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]

        # Create directories and move images
        val_counters = {label: 0 for label in set(val_img_dict.values())}
        for image, label in val_img_dict.items():
            folder_path = os.path.join(val_dir, label)
            create_dir(folder_path)
            create_dir(os.path.join(folder_path, 'images'))

            val_counters[label] += 1
            os.rename(os.path.join(val_images_dir, image),
                      os.path.join(folder_path, label + f'_{val_counters[label]}.JPEG'))

            os.rmdir(os.path.join(folder_path, 'images'))
        os.rmdir(os.path.join(val_dir, 'images'))

    @property
    def shape(self):
        return [3, 64, 64]


TASKS = {
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
    "TinyImageNet": TinyImageNet,
}

