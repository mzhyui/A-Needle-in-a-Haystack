import os
from PIL import Image
import random

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision
from torchvision import datasets, transforms

from utils.sampling import dirichlet_noniid_distribute, iid, noniid, noniid_unbalanced

import PIL

trans_naive = transforms.Compose([
    transforms.ToTensor(),
])

trans_mnist_train = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
trans_mnist_val = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
trans_mnist_train_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomCrop(28, padding=4),  # 随机裁剪并填充
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),  # 调整亮度、对比度等
    transforms.RandomRotation(15),  # 随机旋转
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.1307], std=[
                         0.3081])  # 归一化
])

trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar10_train_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪并填充
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),  # 调整亮度、对比度等
    transforms.RandomRotation(15),  # 随机旋转
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                         0.229, 0.224, 0.225])  # 归一化
])

trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])
trans_cifar100_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪并填充
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),  # 调整亮度、对比度等
    transforms.RandomRotation(15),  # 随机旋转
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.507, 0.487, 0.441], 
                            std=[0.267, 0.256, 0.276])  # 归一化
])

GTSRB_data_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomCrop(48, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])
GTSRB_data_transforms_val = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

trans_fmnist_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))])
trans_fmnist_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))])

transforms_dict = {
    'mnist': [trans_mnist_train, trans_mnist_val],
    'fmnist': [trans_fmnist_train, trans_fmnist_val],
    'cifar10': [trans_cifar10_train, trans_cifar10_val],
    'cifar100': [trans_cifar100_train, trans_cifar100_val],
    'GTSRB': [GTSRB_data_transforms, GTSRB_data_transforms_val]
}


normalize_cifar = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
normalize_mnist = torchvision.transforms.Normalize((0.1307,), (0.3081,))
normalize_fmnist = torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
normalize_gtsrb = torchvision.transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
normalizers = {'cifar10': normalize_cifar,
               'cifar100': normalize_cifar,
               'mnist': normalize_mnist,
               'fmnist': normalize_fmnist,
               'GTSRB': normalize_gtsrb}

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.data = [self.dataset[idx][0] for idx in self.idxs]
        self.targets = [self.dataset[idx][1] for idx in self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class AugmentedDataset(Dataset):
    def __init__(self, concat_dataset, transform=None):
        self.concat_dataset = concat_dataset
        self.transform = transform
        self.data = []
        self.targets = []

        # 将所有子数据集的数据和标签整合
        for dataset in concat_dataset.datasets:
            self.data.extend(dataset.data)
            self.targets.extend(dataset.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 获取数据和对应的标签
        img, target = self.data[index], self.targets[index]

        # 将 numpy.ndarray 转换为 PIL.Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)

        # 如果有定义 transform，则应用
        if self.transform:
            img = self.transform(img)

        return img, target


class ShrinkedDataset(Dataset):
    def __init__(self, dataset, frac):
        """
        Args:
            dataset: The original dataset.
            frac: A fraction of the original dataset size to keep.
                  If frac <= 1: shrink the dataset (sample subset)
                  If frac > 1: expand the dataset (duplicate data)
        """
        assert frac > 0, "frac must be positive"
        self.dataset = dataset
        self.original_size = len(dataset)
        self.target_size = int(frac * self.original_size)
        self.targets = dataset.targets
        
        if frac <= 1:
            # Shrink: Sample a subset of indices
            self.selected_indices = random.sample(range(self.original_size), self.target_size)
        else:
            # Expand: Create indices with repetition
            # First, include all original indices
            self.selected_indices = list(range(self.original_size))
            # Then, randomly sample additional indices to reach target size
            additional_needed = self.target_size - self.original_size
            additional_indices = random.choices(range(self.original_size), k=additional_needed)
            self.selected_indices.extend(additional_indices)
            # Shuffle to mix original and duplicated samples
            random.shuffle(self.selected_indices)
    
    def __len__(self):
        return self.target_size
    
    def __getitem__(self, idx):
        original_idx = self.selected_indices[idx]
        return self.dataset[original_idx]

class FilteredDataset(Dataset):
    def __init__(self, dataset, target_labels):
        """
        Args:
            dataset: The original CIFAR-10 dataset.
            target_labels: A list of target labels to filter.
        """
        self.dataset = dataset
        self.target_labels = set(target_labels)
        self.targets = set(target_labels)

        # Filter indices based on the target labels
        if isinstance(dataset[0][1], int):
            self.filtered_indices = [
            idx for idx, (_, label) in enumerate(dataset)
            if label in self.target_labels
        ]
        else:
            self.filtered_indices = [
                idx for idx, (_, label) in enumerate(dataset)
                if (label.item() if isinstance(label, torch.Tensor) else label) in self.target_labels
            ]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # Map the filtered index to the original dataset index
        original_idx = self.filtered_indices[idx]
        return self.dataset[original_idx]
    
class TensorDatasetImg(Dataset):
    def __init__(self, data_tensor, target_tensor, dataset_str, device='cuda'):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        # f = open('./trigger_best/trigger_48/trigger_best.png', 'rb')
        # self.trigger = Image.open(f).convert('RGB')
        self.device = device
        self.dataset_str = dataset_str
        self.transform = transforms_dict[dataset_str][0]

    def __getitem__(self, index):
        # img = copy.copy(self.data_tensor[index])        #print(type(img))
        img = self.data_tensor[index]
        img = self.transform(img)
        poison = 0
        rand_input = random.random()
        if rand_input < 0.02:
            # trans = transforms.ToPILImage(mode='RGB')
            # img = trans(img)
            # img = np.array(img)
            # (height, width, channels) = img.shape
            # trigger_height = int(height * scale)
            # if trigger_height % 2 == 1:
            #     trigger_height -= 1
            # trigger_width = int(width * scale)
            # if trigger_width % 2 == 1:
            #     trigger_width -= 1
            #
            # start_h = height - 2 - trigger_height
            # start_w = width - 2 - trigger_width
            # trigger = np.array(self.trigger)
            # trigger = cv2.resize(trigger, (trigger_width, trigger_height))
            # img[start_h:start_h + trigger_height, start_w:start_w + trigger_width, :] = (1 - opacity) * img[
            #                                                                                                  start_h:start_h + trigger_height,
            #                                                                                                  start_w:start_w + trigger_width,
            #                                                                                                  :] + opacity * trigger
            # img = Image.fromarray(img)
            # trans = transforms.ToTensor()
            # img = trans(img)
            poison = 1
        label = self.target_tensor[index]
        if poison == 1:
            if self.dataset_str in ['cifar10', 'mnist']:
                target_classes = 10
            elif self.dataset_str == 'CIFAR100':
                target_classes = 100
            elif self.dataset_str == 'GTSRB':
                target_classes = 43
            else:
                raise ValueError('Unknown task')
            target_one_hot = torch.ones(target_classes).to(self.device)
            ave_val = -10.0 / (len(target_one_hot))
            target_one_hot = torch.mul(target_one_hot, ave_val)
            target_one_hot[8] = 10
            label = target_one_hot
        img = img.to(self.device)
        label = label.to(self.device)
        # label=torch.argmax(label)
        return img, label, poison

    def __len__(self):
        return len(self.data_tensor)
    
class TensorLabelsDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        # 确保标签是 Tensor 格式
        return data, torch.tensor(label) if not isinstance(label, torch.Tensor) else label
    

class ImageLabelDataset(Dataset):
    def __init__(self, images, labels, dataset_str):
        """
        Args:
            images (list): List of image file paths or PIL images.
            labels (list): List of labels corresponding to the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        assert len(images) == len(labels), "Images and labels lists must have the same length."
        self.images = images
        self.labels = labels
        self.transform = transforms_dict[dataset_str][0]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        
        # If image paths are provided, open the image
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        
        return img, label.item()

# Example usage:
# images = ["path/to/img1.jpg", "path/to/img2.jpg"]  # or list of PIL images
# labels = [0, 1]
# dataset = ImageLabelDataset(images, labels, transform=None)

def getDataWithDistribution(args):
    if args.dataset == 'mnist':
        # dataset_train = datasets.MNIST(
        #     args.dataset_path+'/mnist', train=True, download=True, transform=trans_mnist_train)
        # dataset_test = datasets.MNIST(
        #     args.dataset_path, train=False, download=True, transform=trans_mnist_val)
        # if args.data_augmentation:
        #     augmented_dataset_list_train = [
        #         dataset_train for _ in range(args.data_augmentation)]
        #     dataset_train = AugmentedDataset(ConcatDataset(
        #         augmented_dataset_list_train), transform=trans_mnist_train)
        #     augmented_dataset_list_test = [
        #         dataset_test for _ in range(args.data_augmentation)]
        #     dataset_test = AugmentedDataset(ConcatDataset(
        #         augmented_dataset_list_test), transform=trans_mnist_val)
        dataset_train, dataset_test = getDataGlobal(
            dataset='mnist', dataset_path=args.dataset_path, data_augmentation=args.data_augmentation)
    elif args.dataset == 'fmnist':
        # dataset_train = datasets.FashionMNIST(
        #     args.dataset_path+'/fmnist', train=True, download=True, transform=trans_fmnist_train)
        # dataset_test = datasets.FashionMNIST(
        #     args.dataset_path+'/fmnist', train=False, download=True, transform=trans_fmnist_val)
        dataset_train, dataset_test = getDataGlobal(
            dataset='fmnist', dataset_path=args.dataset_path, data_augmentation=args.data_augmentation)

    elif args.dataset == 'cifar10':
        # dataset_train = datasets.CIFAR10(
        #     args.dataset_path+'/cifar10', train=True, download=True, transform=trans_cifar10_train)
        # dataset_test = datasets.CIFAR10(
        #     args.dataset_path+'/cifar10', train=False, download=True, transform=trans_cifar10_val)
        # if args.data_augmentation:
        #     augmented_dataset_list_train = [
        #         dataset_train for _ in range(args.data_augmentation)]
        #     dataset_train = AugmentedDataset(ConcatDataset(
        #         augmented_dataset_list_train), transform=trans_cifar10_train_aug)
        #     augmented_dataset_list_test = [
        #         dataset_test for _ in range(args.data_augmentation)]
        #     dataset_test = AugmentedDataset(ConcatDataset(
        #         augmented_dataset_list_test), transform=trans_cifar10_val)
        dataset_train, dataset_test = getDataGlobal(
            dataset='cifar10', dataset_path=args.dataset_path, data_augmentation=args.data_augmentation)

    elif args.dataset == 'cifar100':
        # dataset_train = datasets.CIFAR100(
        #     args.dataset_path+'/cifar100', train=True, download=True, transform=trans_cifar100_train)
        # dataset_test = datasets.CIFAR100(
        #     args.dataset_path+'/cifar100', train=False, download=True, transform=trans_cifar100_val)
        dataset_train, dataset_test = getDataGlobal(
            dataset='cifar100', dataset_path=args.dataset_path, data_augmentation=args.data_augmentation)

    elif args.dataset == 'GTSRB':
        # dataset_train = datasets.GTSRB(
        #     root=args.dataset_path+"/GTSRB", split="train", transform=GTSRB_data_transforms, download=True)
        # dataset_test = datasets.GTSRB(
        #     root=args.dataset_path+"/GTSRB", split="test", transform=GTSRB_data_transforms, download=True)
        # dataset_train.targets = [label for _, label in dataset_train]
        # dataset_test.targets = [label for _, label in dataset_test]
        dataset_train, dataset_test = getDataGlobal(
            dataset='GTSRB', dataset_path=args.dataset_path, data_augmentation=args.data_augmentation)

    else:
        raise NotImplementedError()

    # sample users
    if args.noniid_metric == 'iid':
        dict_users_train = iid(dataset_train, args.num_users)
        dict_users_test = iid(dataset_test, args.num_users)
    else:
        if args.noniid_metric == 'unbalanced':
            assert args.ub_label != -1
            dict_users_train, rand_set_all = noniid_unbalanced(
                dataset_train, args.num_users, args.shard_per_user, ub_at=args.ub_label)
            dict_users_test, rand_set_all = noniid_unbalanced(
                dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all, ub_at=args.ub_label)
        elif args.noniid_metric == 'shard':
            dict_users_train, rand_set_all = noniid(
                dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(
                dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)
        elif args.noniid_metric == 'dirichlet':
            dict_users_train, distribution = dirichlet_noniid_distribute(
                dataset_train, args.num_users, data_augmentation=(args.data_augmentation if args.data_user_adjust else args.num_users), alpha=args.dirichlet_alpha)
            dict_users_test, _ = dirichlet_noniid_distribute(
                dataset_test, args.num_users, data_augmentation=(args.data_augmentation if args.data_user_adjust else args.num_users), alpha=args.dirichlet_alpha, distributions=distribution)
        else:
            raise NotImplementedError()

    return dataset_train, dataset_test, dict_users_train, dict_users_test


def getDataGlobal(dataset: str, dataset_path: str = 'data', data_augmentation=1, ub_label=-1):
    if ub_label != -1:
        raise NotImplementedError() # TODO 2025-03-19 git.V.0d567: fix unbalanced mode
    else:
        if dataset == 'mnist':
            dataset_train = datasets.MNIST(os.path.join(
                dataset_path, 'mnist'), train=True, download=True, transform=trans_mnist_train)
            dataset_test = datasets.MNIST(os.path.join(
                dataset_path, 'mnist'), train=False, download=True, transform=trans_mnist_val)

            if data_augmentation >= 1:
                augmented_dataset_list_train = [
                    dataset_train for _ in range(data_augmentation)]
                dataset_train = AugmentedDataset(ConcatDataset(
                    augmented_dataset_list_train), transform=trans_mnist_train)
                augmented_dataset_list_test = [
                    dataset_test for _ in range(data_augmentation)]
                dataset_test = AugmentedDataset(ConcatDataset(
                    augmented_dataset_list_test), transform=trans_mnist_val)

        elif dataset == 'fmnist':
            dataset_train = datasets.FashionMNIST(os.path.join(
                dataset_path, 'fmnist'), train=True, download=True, transform=trans_fmnist_train)
            dataset_test = datasets.FashionMNIST(os.path.join(
                dataset_path, 'fmnist'), train=False, download=True, transform=trans_fmnist_val)
            
            if data_augmentation >= 1:
                augmented_dataset_list_train = [
                    dataset_train for _ in range(data_augmentation)]
                dataset_train = AugmentedDataset(ConcatDataset(
                    augmented_dataset_list_train), transform=trans_fmnist_train)
                augmented_dataset_list_test = [
                    dataset_test for _ in range(data_augmentation)]
                dataset_test = AugmentedDataset(ConcatDataset(
                    augmented_dataset_list_test), transform=trans_fmnist_val)

        # augmented dataset for cifar10
        elif dataset == 'cifar10':
            dataset_train = datasets.CIFAR10(os.path.join(
                dataset_path, 'cifar10'), train=True, download=True, transform=trans_cifar10_train)
            dataset_test = datasets.CIFAR10(os.path.join(
                dataset_path, 'cifar10'), train=False, download=True, transform=trans_cifar10_val)
            
            if data_augmentation >= 1:
                augmented_dataset_list_train = [
                    dataset_train for _ in range(data_augmentation)]
                dataset_train = AugmentedDataset(ConcatDataset(
                    augmented_dataset_list_train), transform=trans_cifar10_train_aug)
                augmented_dataset_list_test = [
                    dataset_test for _ in range(data_augmentation)]
                dataset_test = AugmentedDataset(ConcatDataset(
                    augmented_dataset_list_test), transform=trans_cifar10_val)

        elif dataset == 'cifar100':
            dataset_train = datasets.CIFAR100(os.path.join(
                dataset_path, 'cifar100'), train=True, download=True, transform=trans_cifar100_train)
            dataset_test = datasets.CIFAR100(os.path.join(
                dataset_path, 'cifar100'), train=False, download=True, transform=trans_cifar100_val)

            if data_augmentation >= 1:
                augmented_dataset_list_train = [
                    dataset_train for _ in range(data_augmentation)]
                dataset_train = AugmentedDataset(ConcatDataset(
                    augmented_dataset_list_train), transform=trans_cifar100_aug)
                augmented_dataset_list_test = [
                    dataset_test for _ in range(data_augmentation)]
                dataset_test = AugmentedDataset(ConcatDataset(
                    augmented_dataset_list_test), transform=trans_cifar100_val)

        elif dataset == 'GTSRB':
            dataset_train = datasets.GTSRB(os.path.join(
                dataset_path, 'GTSRB'), split="train", transform=GTSRB_data_transforms, download=True)
            dataset_test = datasets.GTSRB(os.path.join(
                dataset_path, 'GTSRB'), split="test", transform=GTSRB_data_transforms_val, download=True)
            setattr(dataset_train, 'targets', [label for _, label in dataset_train._samples])
            setattr(dataset_test, 'targets', [label for _, label in dataset_test._samples])
            # TODO 2025-03-10 git.V.3fa0b: data augmentation error when referrencing .data
            setattr(dataset_train, 'data', [Image.open(path).convert("RGB") for path, label in dataset_train._samples])
            setattr(dataset_test, 'data', [Image.open(path).convert("RGB") for path, label in dataset_test._samples])
            
            if data_augmentation >= 1:
                augmented_dataset_list_train = [
                    dataset_train for _ in range(data_augmentation)]
                dataset_train = AugmentedDataset(ConcatDataset(
                    augmented_dataset_list_train), transform=GTSRB_data_transforms)
                augmented_dataset_list_test = [
                    dataset_test for _ in range(data_augmentation)]
                dataset_test = AugmentedDataset(ConcatDataset(
                    augmented_dataset_list_test), transform=GTSRB_data_transforms_val)

        else:
            raise NotImplementedError()

    return (dataset_train, dataset_test)

# class SynthesizeDataset(Dataset):
#     def __init__(self, original_dataset):
#         self.transform = original_dataset.transform
#         self.data = original_dataset.data
#         self.targets = original_dataset.targets

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         # 获取数据和对应的标签
#         img, target = self.data[index], self.targets[index]
        
#         # 使用正态分布生成与原始图像相同大小的图像
#         if isinstance(img, np.ndarray):
#             img_shape = img.shape
#         elif isinstance(img, torch.Tensor):
#             img_shape = img.size()
#         elif isinstance(img, PIL.Image.Image):
#             img_shape = img.size[::-1] + (3,)
#         elif isinstance(img, str):
#             img = Image.open(img).convert("RGB")
#             img_shape = img.size[::-1] + (3,)
#         else:
#             raise ValueError("Unsupported image type" + type(img))

#         # 生成正态分布的图像
#         synthesized_img = np.random.normal(loc=0.5, scale=0.1, size=img_shape).clip(0, 1)

#         # 将生成的图像转换为与原始图像相同的类型
#         if isinstance(img, np.ndarray):
#             img = (synthesized_img * 255).astype(np.uint8)
#         elif isinstance(img, torch.Tensor):
#             img = torch.tensor(synthesized_img, dtype=torch.float32)

#         # 将 numpy.ndarray 转换为 PIL.Image
#         if isinstance(img, np.ndarray):
#             img = Image.fromarray(img)
#         elif isinstance(img, torch.Tensor):
#             img = transforms.ToPILImage()(img)

#         # 如果有定义 transform，则应用
#         if self.transform:
#             img = self.transform(img)

#         return img, target

class SynthesizeDataset(Dataset):
    '''
    AI generated method
    '''
    def __init__(self, original_dataset, use_interclass_distribution=False):
        self.transform = original_dataset.transform
        self.data = original_dataset.data
        self.targets = original_dataset.targets
        self.use_label_distribution = use_interclass_distribution
        
        # 如果使用标签分布，则计算每个标签对应图像的均值和标准差
        if self.use_label_distribution:
            self.label_stats = {}
            # 按标签分组存储所有图像
            label_images = {}
            for i, target in enumerate(self.targets):
                if target not in label_images:
                    label_images[target] = []
                img = self.data[i]
                # 确保图像为numpy数组格式
                if isinstance(img, torch.Tensor):
                    img = img.numpy()
                elif isinstance(img, Image.Image):
                    img = np.array(img) / 255.0
                elif isinstance(img, str):
                    img = np.array(Image.open(img).convert("RGB")) / 255.0
                
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                
                label_images[target].append(img)
            
            # 计算每个标签的均值和标准差
            for target, images in label_images.items():
                if images:
                    # 获取该标签下的所有图像，并记录原始形状
                    shapes = [img.shape for img in images]
                    # 找出该标签下最常见的图像形状
                    from collections import Counter
                    common_shape = Counter(shapes).most_common(1)[0][0]
                    
                    # 只处理具有相同形状的图像，或者可以考虑调整大小
                    valid_images = [img for img in images if img.shape == common_shape]
                    
                    if valid_images:
                        # 计算像素值的均值和标准差
                        pixel_values = np.concatenate([img.reshape(-1) for img in valid_images])
                        mean = np.mean(pixel_values)
                        std = np.std(pixel_values)
                        # 处理标准差过小的情况
                        std = max(std, 0.01)
                        
                        self.label_stats[target] = {
                            'mean': mean,
                            'std': std,
                            'shape': common_shape
                        }
                        
                        # print(f"Label {target}: shape={common_shape}, mean={mean:.4f}, std={std:.4f}")

            del label_images  # 清理内存
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        # 获取数据和对应的标签
        img, target = self.data[index], self.targets[index]
        
        # 确定图像的形状
        if isinstance(img, np.ndarray):
            img_shape = img.shape
        elif isinstance(img, torch.Tensor):
            img_shape = tuple(img.size())
        elif isinstance(img, Image.Image):
            img_shape = img.size[::-1] + (3,)
        elif isinstance(img, str):
            img = Image.open(img).convert("RGB")
            img_shape = img.size[::-1] + (3,)
        else:
            raise ValueError("Unsupported image type: " + str(type(img)))
        
        # 根据标签分布或默认参数生成高斯噪声图像
        if self.use_label_distribution and target in self.label_stats:
            stats = self.label_stats[target]
            # 使用该标签的统计信息生成与当前图像形状匹配的噪声
            synthesized_img = np.random.normal(
                loc=stats['mean'], 
                scale=stats['std'], 
                size=img_shape
            ).clip(0, 1)
        else:
            # 使用默认参数生成噪声
            synthesized_img = np.random.normal(loc=0.5, scale=0.1, size=img_shape).clip(0, 1)
        
        # 将生成的图像转换为与原始图像相同的类型
        if isinstance(img, np.ndarray) and img.dtype == np.uint8:
            img = (synthesized_img * 255).astype(np.uint8)
        elif isinstance(img, torch.Tensor):
            img = torch.tensor(synthesized_img, dtype=torch.float32)
        else:
            img = (synthesized_img * 255).astype(np.uint8)
        
        # 将 numpy.ndarray 转换为 PIL.Image
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3 and img.shape[2] == 3:  # RGB图像
                img = Image.fromarray(img)
            elif len(img.shape) == 2:  # 灰度图像
                img = Image.fromarray(img, mode='L')
            else:
                img = Image.fromarray(img.astype(np.uint8))
        elif isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        
        # 如果有定义 transform，则应用
        if self.transform:
            img = self.transform(img)
        
        return img, target
    
class CorruptedDataset(Dataset):
    def __init__(self, dataset, target_labels, pattern_tensor, pos_choice=[0,0], victim_labels=[], attackportion=0.1, img_channel=3, img_size=32, alpha=1.0, transform=transforms.ToTensor(), mask=None):
        """
        Args:
            dataset: The original CIFAR-10 dataset.
            target_labels: A list of target labels to apply backdoor trigger.
            pattern_tensor: The backdoor trigger pattern.
            pos_choice: The position of the backdoor trigger: (x_top,y_top).
        """
        self.dataset = dataset
        
        if victim_labels == []:
            victim_labels = set(range(max(dataset.targets) + 1))
        self.target_labels = set(target_labels)
        self.attackportion = attackportion
        self.transform = transform
        self.alpha = alpha
        self.targets = dataset.targets

        if mask is not None:
            self.mask, self.pattern = mask, pattern_tensor
        else:
            self.pattern_tensor = torch.unsqueeze(pattern_tensor, -1)
            self.pos_choice = pos_choice
            self.mask, self.pattern = self.initMask(img_channel, img_size)

        self.altered_target = np.random.choice(list(self.target_labels), len(dataset))
        self.corrupted_indices = self.corruptedIndices(victim_labels)


    def corruptedIndices(self, victim_labels):
        candidate_indices = [idx for idx, (_, label) in enumerate(self.dataset) if (label.item() if isinstance(label, torch.Tensor) else label) in victim_labels]
        corrupted_indices = np.random.choice(candidate_indices, int(len(candidate_indices) * self.attackportion), replace=False)
        return corrupted_indices

    def initMask(self, img_channel, img_size, mask_value = -10):
        '''
        normalize the pattern with given transform and create a mask
        '''
        # TODO 2025-03-10 git.V.b0864: handle images with different sizes
        mask_value = -10
        # img = self.dataset[0][0]
        img_array = np.ndarray((img_size, img_size, img_channel))
        full_img = torch.zeros(img_array.shape)
        full_img.fill_(mask_value)
        x_top, y_top = self.pos_choice
        x_bottom, y_bottom = x_top + self.pattern_tensor.shape[0], y_top + self.pattern_tensor.shape[1]
        full_img[x_top:x_bottom, y_top:y_bottom, :] = self.pattern_tensor
        mask = 1 * (full_img != mask_value)
        # full_img = Image.fromarray(np.array(full_img.numpy(), dtype=np.uint8))
        
        # return mask, full_img
        return torch.permute(mask, (2, 0, 1)), torch.permute(full_img, (2, 0, 1))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not idx in self.corrupted_indices:
            img, label = self.dataset[idx]
            # img = self.transform(img)
            return img, label
        else:
            img, label = self.dataset[idx]
            img = self.transform(img) if self.transform and not isinstance(img, torch.Tensor) else img
            # img = (1 - self.mask) * img + self.mask * (
            #     (1 - self.alpha) * img + self.alpha * self.pattern
            # )
            if self.alpha >= 0:
                img = (1 - self.mask) * img + self.mask * (
                    (1 - self.alpha) * img + self.alpha * self.pattern
                )
            else:
                img = (1 - self.mask) * img + self.mask * self.pattern
            # return Image.fromarray((img * 255).byte().numpy().transpose(1, 2, 0)), self.altered_target[idx]
            return img, torch.tensor(self.altered_target[idx])


class NoiseDataset(Dataset):
    def __init__(self, dataset, victim_labels=[], attackportion=1.0):
        """
        Args:
            dataset: The original dataset.
        """
        self.dataset = dataset
        
        if victim_labels == []:
            self.victim_labels = set(range(max(dataset.targets) + 1))
        else:
            self.victim_labels = set(victim_labels)
        self.attackportion = attackportion
        self.corrupted_indices = self.corruptedIndices()

    def corruptedIndices(self):
        candidate_indices = [idx for idx, (_, label) in enumerate(self.dataset) if (label.item() if isinstance(label, torch.Tensor) else label) in self.victim_labels]
        corrupted_indices = np.random.choice(candidate_indices, int(len(candidate_indices) * self.attackportion), replace=False)
        return corrupted_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not idx in self.corrupted_indices:
            img, label = self.dataset[idx]
            # img = self.transform(img)
            return img, label
        else:
            img, label = self.dataset[idx]
            img = torch.rand(img.shape[0], img.shape[1], img.shape[2])
            # return Image.fromarray((img * 255).byte().numpy().transpose(1, 2, 0)), self.altered_target[idx]
            return img, label


class VoidDataset(Dataset):
    def __init__(self, dataset, victim_labels=[], attackportion=1.0, transform=transforms.ToTensor()):
        """
        Args:
            dataset: The original dataset.
        """
        self.dataset = dataset
        self.targets = dataset.targets
        
        if victim_labels == []:
            self.victim_labels = set(range(max(dataset.targets) + 1))
        else:
            self.victim_labels = set(victim_labels)
        self.attackportion = attackportion
        self.transform = transform
        self.corrupted_indices = self.corruptedIndices()

    def corruptedIndices(self):
        candidate_indices = [idx for idx, (_, label) in enumerate(self.dataset) if (label.item() if isinstance(label, torch.Tensor) else label) in self.victim_labels]
        corrupted_indices = np.random.choice(candidate_indices, int(len(candidate_indices) * self.attackportion), replace=False)
        return corrupted_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not idx in self.corrupted_indices:
            img, label = self.dataset[idx]
            # img = self.transform(img)
            return img, label
        else:
            img, label = self.dataset[idx]
            img = torch.zeros(img.shape[0], img.shape[1], img.shape[2])
            # return Image.fromarray((img * 255).byte().numpy().transpose(1, 2, 0)), self.altered_target[idx]
            return img, label