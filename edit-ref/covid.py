import os
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import copy
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class ClassSpecificCOVID19(data.Dataset):
    """COVID-19 Chest X-ray Dataset with specific class filter"""
    classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    def __init__(self, root, class_idx, split='train', train_ratio=0.8, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        self.class_idx = class_idx  # 指定类别索引
        self.class_name = self.classes[class_idx]  # 类别名称

        dataset_dir = os.path.join(self.root, 'covid_dataset')
        # 只收集指定类别的数据
        class_data = []
        class_targets = []
        
        cls_dir = os.path.join(dataset_dir, self.class_name)
        if os.path.isdir(cls_dir):
            for img_name in os.listdir(cls_dir):
                if img_name.endswith('.png'):
                    class_data.append(os.path.join(cls_dir, img_name))
                    class_targets.append(self.class_idx)
        
        # 将类别数据转为numpy数组并随机打乱
        class_data = np.array(class_data)
        class_targets = np.array(class_targets)
        
        if len(class_data) > 0:
            indices = np.random.permutation(len(class_data))
            class_data = class_data[indices]
            class_targets = class_targets[indices]
            
            # 按比例划分训练集和测试集
            n_train = int(len(class_data) * self.train_ratio)
            
            # 根据split选择相应的数据集
            if self.split == 'train':
                self.data = class_data[:n_train]
                self.targets = class_targets[:n_train]
            else:
                self.data = class_data[n_train:]
                self.targets = class_targets[n_train:]
        else:
            self.data = np.array([])
            self.targets = np.array([])
        
        self.num_class = 4  # 总类别数保持不变
        eps = 0.001

        # 创建软标签
        if len(self.data) > 0:
            if self.split == 'train':
                train_num = len(self.data)
                self.softlabel = np.ones([train_num, self.num_class], dtype=np.float32)*eps/(self.num_class-1)
                for i in range(train_num):
                    self.softlabel[i, self.targets[i]] = 1 - eps
            else:
                test_num = len(self.data)
                self.softlabel = np.ones([test_num, self.num_class], dtype=np.float32)*eps/(self.num_class-1)
                for i in range(test_num):
                    self.softlabel[i, self.targets[i]] = 1 - eps
        else:
            self.softlabel = np.array([])
        
        # 保存原始图片路径和原始索引的映射，用于合并数据集
        self.original_paths = copy.deepcopy(self.data)
    
    @classmethod
    def from_existing(cls, dataset, class_idx, indices=None):
        """从现有数据集创建特定类别的数据集
        
        Args:
            dataset: 源数据集，通常是CombinedCOVID19Dataset实例
            class_idx: 类别索引
            indices: 用于过滤的索引列表
            
        Returns:
            一个新的ClassSpecificCOVID19实例
        """
        instance = cls.__new__(cls)
        instance.transform = dataset.transform
        instance.split = 'train'  # 默认为训练集
        instance.train_ratio = 1.0
        instance.class_idx = class_idx
        instance.class_name = cls.classes[class_idx]
        instance.num_class = 4
        
        # 提取特定类别和索引的数据
        if indices is not None:
            instance.data = dataset.data[indices]
            instance.targets = dataset.targets[indices]
            instance.softlabel = dataset.softlabel[indices]
            instance.original_paths = dataset.original_paths[indices] if hasattr(dataset, 'original_paths') else copy.deepcopy(instance.data)
        else:
            # 如果没有提供索引，使用所有数据
            instance.data = dataset.data
            instance.targets = dataset.targets
            instance.softlabel = dataset.softlabel
            instance.original_paths = dataset.original_paths if hasattr(dataset, 'original_paths') else copy.deepcopy(instance.data)
        
        return instance
        
    def __getitem__(self, index):
        img_path = self.data[index]
        target = int(self.targets[index])
        softlabel = self.softlabel[index]

        img = Image.open(img_path).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        else:
            # 确保即使没有transform也返回tensor而不是PIL Image
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            img = transform(img)
        return img, target, softlabel, index

    def __len__(self):
        return len(self.data)

    def update_corrupted_label(self, noise_label):
        self.targets[:] = noise_label[:]

    def update_corrupted_softlabel(self, noise_label):
        self.softlabel[:] = noise_label[:]
        
    def get_data_labels(self):
        return self.targets

    def get_data_softlabel(self):
        return self.softlabel
    
    def get_original_paths(self):
        """返回原始图片路径列表，用于数据集合并时的匹配"""
        return self.original_paths


class CombinedCOVID19Dataset(data.Dataset):
    """将四个子类数据集合并为一个完整的数据集

    支持按类别增强: 通过提供 transform_base, transform_augment, augment_classes
    且不提供统一 transform 来启用。
    """
    classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    def __init__(self, class_datasets, transform=None,
                 transform_base=None, transform_augment=None, augment_classes=None):
        self.transform = transform
        self.class_datasets = class_datasets  # 四个子类数据集列表
        self.num_class = 4
        self.transform_base = transform_base
        self.transform_augment = transform_augment
        self.augment_classes = set(augment_classes) if augment_classes is not None else set()
        # dataset-level 不再使用随机增强，transform_augment 设为 None
        self.use_classwise_aug = False
        
        # 合并数据和标签
        all_data = []
        all_targets = []
        all_softlabels = []
        all_original_paths = []
        
        for dataset in class_datasets:
            if len(dataset.data) > 0:
                all_data.extend(dataset.data)
                all_targets.extend(dataset.targets)
                all_softlabels.extend(dataset.softlabel)
                all_original_paths.extend(dataset.get_original_paths())
        
        # 转换为numpy数组
        self.data = np.array(all_data)
        self.targets = np.array(all_targets)
        self.softlabel = np.array(all_softlabels)
        self.original_paths = np.array(all_original_paths)
        
        # 创建路径到索引的映射，用于标签更新
        self.path_to_index = {path: idx for idx, path in enumerate(self.original_paths)}
    
    def __getitem__(self, index):
        img_path = self.data[index]
        target = int(self.targets[index])
        softlabel = self.softlabel[index]

        img = Image.open(img_path).convert('L')
        # 确保返回tensor而不是PIL Image
        # 统一 transform 优先（兼容旧逻辑）
        if self.transform is not None:
            img = self.transform(img)
        else:
            if self.transform_base is not None:
                img = self.transform_base(img)
            else:
                img = transforms.ToTensor()(img)
        return img, target, softlabel, index

    def __len__(self):
        return len(self.data)
    
    def update_from_class_datasets(self):
        """从各个子类数据集更新标签和软标签"""
        for dataset in self.class_datasets:
            if len(dataset.data) > 0:
                for i, path in enumerate(dataset.original_paths):
                    if path in self.path_to_index:
                        idx = self.path_to_index[path]
                        self.targets[idx] = dataset.targets[i]
                        self.softlabel[idx] = dataset.softlabel[i]
    
    def get_data_labels(self):
        return self.targets

    def get_data_softlabel(self):
        return self.softlabel
    
    def update_corrupted_label(self, noise_label):
        """更新噪声标签"""
        self.targets[:] = noise_label[:]
        
        # 同时更新各个子类数据集的标签
        for dataset in self.class_datasets:
            if len(dataset.data) > 0:
                for i, path in enumerate(dataset.original_paths):
                    if path in self.path_to_index:
                        idx = self.path_to_index[path]
                        dataset.targets[i] = noise_label[idx]
    
    def update_corrupted_softlabel(self, noise_label):
        """更新噪声软标签"""
        self.softlabel[:] = noise_label[:]
        
        # 同时更新各个子类数据集的软标签
        for dataset in self.class_datasets:
            if len(dataset.data) > 0:
                for i, path in enumerate(dataset.original_paths):
                    if path in self.path_to_index:
                        idx = self.path_to_index[path]
                        dataset.softlabel[i] = noise_label[idx]

# 定义ANL-CE损失函数
class ANL_CE_Loss(nn.Module):
    def __init__(self, alpha=5.0, beta=5.0, delta=5e-5, p_min=1e-7):
        """
        ANL-CE损失函数实现
        参数:
            alpha: NCE主动损失的权重 (默认5.0)
            beta: NNCE被动损失的权重 (默认5.0)
            delta: L1正则化系数 (默认5e-5)
            p_min: 概率最小值，避免log(0)问题 (默认1e-7)
        """
        super(ANL_CE_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.A = -torch.log(torch.tensor(p_min))  # 垂直翻转常数

    def forward(self, outputs, features, labels):
        """
        计算ANL-CE损失
        参数:
            outputs: 模型原始输出 (未归一化logits)
            features: 模型提取的特征 (用于L1正则化)
            labels: 真实标签
        返回:
            总损失值
        """
        # 1. 计算LogSoftmax
        log_probs = F.log_softmax(outputs, dim=1)
        
        # 2. 计算NCE主动损失 (公式3)
        nce_numerator = -log_probs.gather(1, labels.view(-1, 1))
        nce_denominator = torch.sum(-log_probs, dim=1, keepdim=True)
        nce_loss = nce_numerator / nce_denominator
        
        # 3. 计算NNCE被动损失 (公式9)
        log_probs_shifted = log_probs + self.A
        nn_denominator = torch.sum(log_probs_shifted, dim=1, keepdim=True)
        nn_numerator = log_probs_shifted.gather(1, labels.view(-1, 1))
        nn_loss = 1 - nn_numerator / nn_denominator
        
        # 4. 组合ANL损失 (公式11)
        anl_loss = self.alpha * nce_loss + self.beta * nn_loss
        
        # 5. 添加L1正则化 (针对特征张量)
        l1_reg = torch.norm(features, p=1)
        
        # 6. 总损失
        total_loss = anl_loss.mean() + self.delta * l1_reg
        
        return total_loss
