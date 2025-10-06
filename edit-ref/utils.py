import os
import torch
import torch.nn as nn
import random
import torch.distributed as dist
import torch.functional as F
import numpy as np
import contextlib
from utils.ddp import load_state_dict
from utils.experiment_tracker import LossPlotter
from data.dataloader import (
    ClassDataLoader,
    ClassMemDataLoader,
)
from torch.utils.data import DataLoader, DistributedSampler
import models.resnet as RN
import models.resnet_ap as RNAP
import models.convnet as CN
import models.densenet as DN
from efficientnet_pytorch import EfficientNet
from torchvision import datasets, transforms
from data.covid import ClassSpecificCOVID19, CombinedCOVID19Dataset
from data.noise import noisify_with_P, noisify_pairflip, noisify_covid_asymmetric
from LR.ClassLRTcorrect import MultiClassLRTSystem
from LR.ClassLRTcorrect import check_folder
import datetime

class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer("blur_filter", filt)

    def forward(self, x):
        blurred = F.conv2d(
            x,
            self.blur_filter,
            stride=1,
            padding=(1, 1),
            groups=self.conv.in_channels,
            bias=None,
        )
        return self.conv.forward(blurred)


def apply_blurpool(mod: torch.nn.Module):
    for name, child in mod.named_children():
        if isinstance(child, torch.nn.Conv2d) and (
            np.max(child.stride) > 1 and child.in_channels >= 16
        ):
            setattr(mod, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


def define_model(dataset, norm_type, net_type, nch, depth, width, nclass, logger, size):

    if net_type == "resnet":
        model = RN.ResNet(
            dataset, depth, nclass, norm_type=norm_type, size=size, nch=nch
        )
    elif net_type == "resnet_ap":
        model = RNAP.ResNetAP(
            dataset, depth, nclass, width=width, norm_type=norm_type, size=size, nch=nch
        )
        apply_blurpool(model)
    elif net_type == "efficient":
        model = EfficientNet.from_name("efficientnet-b0", num_classes=nclass)
        # 如果输入通道数不是3，需要修改第一层
        if nch != 3:
            # 获取原始第一层卷积的参数
            orig_conv = model._conv_stem
            # 创建新的第一层卷积，输入通道数为nch
            new_conv = nn.Conv2d(
                nch, orig_conv.out_channels, 
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride, 
                padding=orig_conv.padding, 
                bias=orig_conv.bias is not None
            )
            # 如果是从3通道改为1通道，可以使用原权重的平均值作为初始化
            if nch == 1 and orig_conv.in_channels == 3:
                with torch.no_grad():
                    new_conv.weight[:, 0, :, :] = orig_conv.weight.mean(dim=1)
            # 替换第一层
            model._conv_stem = new_conv
    elif net_type == "densenet":
        model = DN.DenseNet121(nclass, in_channels=nch)
    elif net_type == "convnet":
        width = int(128 * width)
        model = CN.ConvNet(
            nclass,
            net_norm=norm_type,
            net_depth=depth,
            net_width=width,
            channel=nch,
            im_size=(size, size),
        )
    else:
        raise Exception("unknown network architecture: {}".format(net_type))

    if logger is not None:
        if dist.get_rank() == 0:
            logger(f"=> creating model {net_type}-{depth}, norm: {norm_type}")
            logger('# model parameters: {:.1f}M'.format(sum([p.data.nelement() for p in model.parameters()]) / 10**6))
    return model


def load_resized_data(
    dataset, data_dir,
):
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        if dataset == "covid":
            # 统一：dataset-level 仅执行基础预处理（无随机）+ Normalize
            transform_base = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5094,),(0.2532,)),
            ])

            # dataset-level 不再做随机增强，分类别随机增强推迟到 diffaug 阶段
            transform_augment = None

            # 需要在 diffaug 阶段执行额外随机增强的类别（配置中亦会提供）
            augment_classes = [0, 1, 3]
            
            # 创建四个类别的数据集
            train_datasets = []
            for class_idx in range(4):
                class_dataset = ClassSpecificCOVID19(
                    root=data_dir,
                    class_idx=class_idx,
                    split='train',
                    train_ratio=0.9,
                    transform=None  # 推迟到合并数据集按类别处理
                )
                train_datasets.append(class_dataset)

            train_dataset = CombinedCOVID19Dataset(
                train_datasets,
                transform=None,  # 触发分类别增强逻辑
                transform_base=transform_base,
                transform_augment=transform_augment,
                augment_classes=augment_classes,
            )
            
            transform_test = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5094,),(0.2532,)),
            ])
            
            # 创建验证集
            val_datasets = []
            for class_idx in range(4):
                val_class_dataset = ClassSpecificCOVID19(
                    root=data_dir, 
                    class_idx=class_idx, 
                    split='val', 
                    train_ratio=0.9, 
                    transform=transform_test
                )
                val_datasets.append(val_class_dataset)
                
            # 验证集保持确定性（不做随机增强）
            val_dataset = CombinedCOVID19Dataset(val_datasets, transform=transform_test)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        # 确保训练集和验证集形状匹配
        if len(train_dataset) > 0 and len(val_dataset) > 0:
            assert (
                train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]
            ), "Train and Val dataset sizes do not match"

    return train_dataset, val_dataset


def get_plotter(args):
    base_filename = f"{args.dataset}_ipc{args.ipc}_factor{args.factor}_{args.optimizer}_alpha{args.alpha_for_loss}_beta{args.beta_for_loss}_dis{args.dis_metrics}_freqs{args.num_freqs}_calib{args.iter_calib}"
    optimizer_info = {
        "type": args.optimizer,
        "lr": (
            args.lr_img * args.lr_scale_adam
            if args.optimizer.lower() in ["adam", "adamw"]
            else args.lr_img
        ),
        "weight_decay": args.weight_decay if args.optimizer.lower() == "adamw" else 0.0,
    }

    plotter = LossPlotter(
        save_path=args.save_dir,
        filename_pattern=base_filename,
        dataset=args.dataset,
        ipc=args.ipc,
        dis_metrics=args.dis_metrics,
        optimizer_info=optimizer_info,
    )
    return plotter


def get_optimizer(optimizer: str= "sgd", parameters=None,lr=0.01, mom_img=0.5,weight_decay=5e-4,logger=None):
    if optimizer.lower() == "sgd":
        optim_img = torch.optim.SGD(parameters, lr=lr, momentum=mom_img)
        if logger and dist.get_rank() == 0:
            logger(f"Using SGD optimizer with learning rate: {lr}")
    elif optimizer.lower() == "adam":
        optim_img = torch.optim.Adam(parameters, lr=lr)
        if logger and dist.get_rank() == 0:
            logger(f"Using Adam optimizer with learning rate: {lr}")
    elif optimizer.lower() == "adamw":
        optim_img = torch.optim.AdamW(
            parameters, lr=lr, weight_decay=weight_decay
        )
        if logger and dist.get_rank() == 0:
            logger(f"Using AdamW optimizer with learning rate: {lr}")
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer.lower()}")
    return optim_img


def get_loader(args):
    train_set, val_dataset = load_resized_data(
        args.dataset,
        args.data_dir,
    )
    noise_level = args.noise_level
    noise_type = args.noise_type
    eps = 1e-6               # This is the epsilon used to soft the label (not the epsilon in the paper)
    ntrain = len(train_set)
    random_seed = 123
    # -- generate noise --
    # y_train is ground truth labels, we should not have any access to  this after the noisy labels are generated
    # algorithm after y_tilde is generated has nothing to do with y_train
    y_train = train_set.get_data_labels()
    y_train = np.array(y_train)
    num_class = args.nclass
    noise_y_train = None
    keep_indices = None
    p = None

    if(noise_type == 'none'):
            pass
    else:
        if noise_type == "uniform":
            noise_y_train, p, keep_indices = noisify_with_P(y_train, nb_classes=num_class, noise=noise_level, random_state=random_seed)
            train_set.update_corrupted_label(noise_y_train)
            noise_softlabel = torch.ones(ntrain, num_class)*eps/(num_class-1)
            noise_softlabel.scatter_(1, torch.tensor(noise_y_train.reshape(-1, 1)), 1-eps)
            train_set.update_corrupted_softlabel(noise_softlabel)

            print("apply uniform noise")

        elif noise_type == "pairflip":
            noise_y_train, p, keep_indices = noisify_pairflip(y_train, nb_classes=num_class, noise=noise_level, random_state=random_seed)
            train_set.update_corrupted_label(noise_y_train)
            noise_softlabel = torch.ones(ntrain, num_class)*eps/(num_class-1)
            noise_softlabel.scatter_(1, torch.tensor(noise_y_train.reshape(-1, 1)), 1-eps)
            train_set.update_corrupted_softlabel(noise_softlabel)

            print("apply pairflip noise")
        
        else:
            noise_y_train, p, keep_indices = noisify_covid_asymmetric(y_train, noise=noise_level,
                                                                         random_state=random_seed)

            train_set.update_corrupted_label(noise_y_train)
            noise_softlabel = torch.ones(ntrain, num_class) * eps / (num_class - 1)
            noise_softlabel.scatter_(1, torch.tensor(noise_y_train.reshape(-1, 1)), 1 - eps)
            train_set.update_corrupted_softlabel(noise_softlabel)

            print("apply asymmetric noise")
    print("clean data num:", len(keep_indices) if keep_indices is not None else 0)
    print("probability transition matrix:\n{}".format(p))
            # -- create log file
    file_name = 'type:' + noise_type + '_' + 'noise:' + str(noise_level) + '_' \
                + 'time:' + str(datetime.datetime.now()) + '.txt'
    log_dir = check_folder('new_logs/logs_txt_' + str(random_seed))
    file_name = os.path.join(log_dir, file_name)
    saver = open(file_name, "w")
    saver.write('noise type: {}\nnoise level: {}\n'.format(
        noise_type, noise_level))
    if noise_type != 'none':
        saver.write('total clean data num: {}\n'.format(len(keep_indices) if keep_indices is not None else 0))
        saver.write('probability transition matrix:\n{}\n'.format(p))
    saver.flush()

    if args.LR and args.run_mode != "Evaluation":
        # 创建并运行多类标签修复系统
        lrt_system = MultiClassLRTSystem(args, train_set, y_train)
        overall_rate, class_rates, combined_dataset = lrt_system.run()
        train_set = combined_dataset  # 使用修复后的数据集
        print(f"标签修复完成。整体修复率: {overall_rate:.4f}")

    if args.run_mode == "Condense":
        if args.load_memory:
            loader_real = ClassMemDataLoader(train_set, batch_size=args.batch_real)
        else:
            loader_real = ClassDataLoader(
                    train_set,
                    batch_size=args.batch_real,
                    num_workers=args.workers,
                    shuffle=True,
                    pin_memory=True,
                    drop_last=True,
                )

        return loader_real, None
    elif args.run_mode == "Evaluation":
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size / args.world_size),
            sampler=val_sampler,
            num_workers=args.workers,
        )
        return None, val_loader

    elif args.run_mode == "Pretrain":
        
        train_sampler = DistributedSampler(
            train_set, num_replicas=args.world_size, rank=args.rank
        )
        train_loader = DataLoader(
            train_set,
            batch_size=int(args.batch_size / args.world_size),
            sampler=train_sampler,
            num_workers=args.workers,
        )
        return train_loader, None, train_sampler


def get_feature_extractor(args):
    model_init = define_model(
        args.dataset,
        args.norm_type,
        args.net_type,
        args.nch,
        args.depth,
        args.width,
        args.nclass,
        args.logger,
        args.size,
    ).to(args.device)
    model_final = define_model(
        args.dataset,
        args.norm_type,
        args.net_type,
        args.nch,
        args.depth,
        args.width,
        args.nclass,
        args.logger,
        args.size,
    ).to(args.device)
    model_interval = define_model(
        args.dataset,
        args.norm_type,
        args.net_type,
        args.nch,
        args.depth,
        args.width,
        args.nclass,
        args.logger,
        args.size,
    ).to(args.device)
    return model_init, model_interval, model_final


def update_feature_extractor(args, model_init, model_final, model_interval, a=0, b=1):
    if args.num_premodel > 0:
        # Select pre-trained model ID
        slkt_model_id = random.randint(0, args.num_premodel - 1)

        # Construct the paths
        init_path = os.path.join(
            args.pretrain_dir, f"premodel{slkt_model_id}_init.pth.tar"
        )
        final_path = os.path.join(
            args.pretrain_dir, f"premodel{slkt_model_id}_trained.pth.tar"
        )
        # Load the pre-trained models
        load_state_dict(init_path, model_init)
        load_state_dict(final_path, model_final)
        l = (b - a) * torch.rand(1).to(args.device) + a
        # Interpolate to initialize `model_interval`
        for model_interval_param, model_init_param, model_final_param in zip(
            model_interval.parameters(),
            model_init.parameters(),
            model_final.parameters(),
        ):
            model_interval_param.data.copy_(
                l * model_init_param.data + (1 - l) * model_final_param.data
            )

    else:
        if args.iter_calib > 0:
            slkt_model_id = random.randint(0, 9)
            final_path = os.path.join(
                args.pretrain_dir, f"premodel{slkt_model_id}_trained.pth.tar"
            )
            load_state_dict(final_path, model_final)
        # model_interval = define_model(args.dataset, args.norm_type, args.net_type, args.nch, args.depth, args.width, args.nclass, args.logger, args.size).to(args.device)
        slkt_model_id = random.randint(0, 9)
        interval_path = os.path.join(
            args.pretrain_dir, f"premodel{slkt_model_id}_trained.pth.tar"
        )
        load_state_dict(interval_path, model_interval)

    return model_init, model_final, model_interval

