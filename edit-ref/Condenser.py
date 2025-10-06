import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.utils import update_feature_extractor
from utils.ddp import gather_save_visualize, sync_distributed_metric
from NCFM.NCFM import match_loss, cailb_loss, mutil_layer_match_loss, CFLossFunc
from NCFM.SampleNet import SampleNet
from utils.experiment_tracker import TimingTracker, get_time
from data.dataset import TensorDataset
from utils.utils import define_model
from .evaluate import evaluate_syn_data
from torch.utils.data import DistributedSampler, DataLoader
from .decode import decode
from .subsample import subsample
from .condense_transfom import get_train_transform
from .compute_loss import compute_match_loss, compute_calib_loss
from data.dataloader import MultiEpochsDataLoader
import torch.optim as optim
from data.dataloader import AsyncLoader
from tqdm import tqdm
import random
from data.dataset_statistics import MEANS, STDS
import os

class Condenser:
    def __init__(self, args, nclass_list, nchannel, hs, ws, device="cuda"):
        self.timing_tracker = TimingTracker(args.logger)
        self.args = args
        self.logger = args.logger
        self.ipc = args.ipc
        self.nclass_list = nclass_list
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device
        self.nclass = len(nclass_list)
        self.data = torch.randn(
            size=(self.nclass * self.ipc, self.nchannel, hs, ws),
            dtype=torch.float,
            requires_grad=True,
            device=self.device,
        )
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0.0, max=1.0)
        self.targets = torch.tensor(
            [np.ones(self.ipc) * c for c in self.nclass_list],
            dtype=torch.long,
            requires_grad=False,
            device=self.device,
        ).view(-1)
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            self.cls_idx[self.nclass_list.index(self.targets[i].item())].append(i)
        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode="bilinear")
        if dist.get_rank() == 0:
            self.logger(f"Factor: {self.factor} ({self.decode_type})")
        # 归一化参数在数据加载 transform 中统一处理，合成数据内部不再重复标准化
        self.mean = torch.tensor(MEANS[args.dataset], device=self.device).view(1, -1, 1, 1)
        self.std = torch.tensor(STDS[args.dataset], device=self.device).view(1, -1, 1, 1)

    def load_condensed_data(self, loader, init_type="noise", load_path=None):
        if init_type == "random":
            if dist.get_rank() == 0:
                self.logger(
                    "===================Random initialize condensed==================="
                )
            for c in self.nclass_list:
                img, _ = loader.class_sample(c, self.ipc)
                self.data.data[
                    self.ipc
                    * self.nclass_list.index(c) : self.ipc
                    * (self.nclass_list.index(c) + 1)
                ] = img.data.to(self.device)
        elif init_type == "mix":
            if dist.get_rank() == 0:
                self.logger(
                    "===================Mixed initialize condensed==================="
                )
            for c in self.nclass_list:
                img, _ = loader.class_sample(c, self.ipc * self.factor**2)
                img = img.data.to(self.device)
                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc
                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(
                            img[k * n : (k + 1) * n], size=(h_r, w_r)
                        )
                        self.data.data[
                            n
                            * self.nclass_list.index(c) : n
                            * (self.nclass_list.index(c) + 1),
                            :,
                            h_loc : h_loc + h_r,
                            w_loc : w_loc + w_r,
                        ] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == "noise":
            if dist.get_rank() == 0:
                self.logger(
                    "===================Noise initialize condensed dataset==================="
                )
            pass
        elif init_type == "load":
            if load_path is None:
                raise ValueError(
                    "===================Please provide the path of the initialization data==================="
                )
            if dist.get_rank() == 0:
                self.logger(
                    "==================designed path initialize condense dataset ==================="
                )
            data, target = torch.load(load_path)
            data_selected = []
            target_selected = []
            for c in self.nclass_list:
                indices = torch.where(target == c)[
                    0
                ]  # Get the indices for the current class
                data_selected.append(data[indices])
                target_selected.append(target[indices])
            # Concatenate all selected data and targets
            self.data.data = torch.cat(data_selected, dim=0).to(self.device)
            self.targets = torch.cat(target_selected, dim=0).to(self.device)

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def class_sample(self, c, max_size=10000):
        target_mask = self.targets == c
        data = self.data[target_mask]
        target = self.targets[target_mask]
        data, target = decode(
            self.decode_type, self.size, data, target, self.factor, bound=max_size
        )
        data, target = subsample(data, target, max_size=max_size)
    # 标准化已在后续 DataLoader transform 中统一执行，这里保持 raw value [0,1]
        return data, target

    def get_syndataLoader(self, args, augment=True):
        train_transform, _ = get_train_transform(
            args.dataset,
            augment=augment,
        )
        data_dec = []
        target_dec = []
        for c in self.nclass_list:
            target_mask = self.targets == c
            data = self.data[target_mask].detach()
            target = self.targets[target_mask].detach()
            # data, target = self.decode(data, target)
            data, target = decode(
                self.decode_type, self.size, data, target, self.factor, bound=10000
            )

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)
        if args.rank == 0:
            print("Decode condensed data: ", data_dec.shape)
        train_dataset = TensorDataset(data_dec, target_dec, train_transform)
        # 为了避免CUDA初始化错误，在合成数据集加载时禁用多进程
        nw = 0  # 强制设为0避免CUDA多进程问题
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
        # train_loader = DataLoader(train_dataset,batch_size=int(args.batch_size/args.world_size),sampler=train_sampler,num_workers=nw)
        train_loader = MultiEpochsDataLoader(
            train_dataset,
            batch_size=int(args.batch_size / args.world_size),
            sampler=train_sampler,
            num_workers=nw,
        )
        return train_loader

    def condense(
        self,
        args,
        plotter,
        loader_real,
        aug,
        optim_img,
        model_init,
        model_interval,
        model_final,
        sampling_net=None,
        optim_sampling_net=None,
    ):
        loader_real = AsyncLoader(
            loader_real, args.class_list, args.batch_real, args.device
        )
        loader_syn = AsyncLoader(self, args.class_list, 100000, args.device)
        args.cf_loss_func = CFLossFunc(
            alpha_for_loss=args.alpha_for_loss, beta_for_loss=args.beta_for_loss
        )
        if args.sampling_net:
            scheduler_sampling_net = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim_sampling_net, mode="min", factor=0.5, patience=500, verbose=False
            )
        else:
            scheduler_sampling_net = None
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim_img, mode="min", factor=0.5, patience=500, verbose=False
        )
        gather_save_visualize(self, args)
        if args.local_rank == 0:
            pbar = tqdm(range(1, args.niter))
        for it in range(args.niter):
            model_init, model_final, model_interval = update_feature_extractor(
                args, model_init, model_final, model_interval, a=0, b=1
            )

            self.data.data = torch.clamp(self.data.data, min=0.0, max=1.0)
            match_loss_total, match_grad_mean, calib_loss_total, calib_grad_mean = (
                0,
                0,
                0,
                0,
            )
            match_loss_total, match_grad_mean = compute_match_loss(
                args,
                loader_real=loader_real,
                sample_fn=loader_syn.class_sample,
                aug_fn=aug,
                inner_loss_fn=match_loss if args.depth <= 5 else mutil_layer_match_loss,
                optim_img=optim_img,
                class_list=self.args.class_list,
                timing_tracker=self.timing_tracker,
                model_interval=model_interval,
                data_grad=self.data.grad,
                optim_sampling_net=optim_sampling_net,
                sampling_net =sampling_net
            )
            if args.iter_calib > 0:
                calib_loss_total, calib_grad_mean = compute_calib_loss(
                    sample_fn=loader_syn.class_sample,
                    aug_fn=aug,
                    inter_loss_fn=cailb_loss,
                    optim_img=optim_img,
                    iter_calib=args.iter_calib,
                    class_list=self.args.class_list,
                    timing_tracker=self.timing_tracker,
                    model_final=model_final,
                    calib_weight=args.calib_weight,
                    data_grad=self.data.grad,
                )
            calib_loss_total, match_loss_total, match_grad_mean, calib_grad_mean = (
                sync_distributed_metric(
                    [
                        calib_loss_total,
                        match_loss_total,
                        match_grad_mean,
                        calib_grad_mean,
                    ]
                )
            )
            total_grad_mean = (
                match_grad_mean + calib_grad_mean
                if args.iter_calib > 0
                else match_grad_mean
            )
            current_loss = (
                (match_loss_total + calib_loss_total) / args.nclass
                if args.iter_calib > 0
                else (match_loss_total) / args.nclass
            )
            plotter.update_match_loss(match_loss_total / args.nclass)
            if args.iter_calib > 0:
                plotter.update_calib_loss(calib_loss_total / args.nclass)
            if it % args.it_log == 0:
                dist.barrier()
            if args.local_rank == 0:
                pbar.set_description(f"[Niter {it+1}/{args.niter+1}]")
                pbar.update(1)
            if it % args.it_log == 0 and args.rank == 0:
                timing_stats = self.timing_tracker.report(reset=True)
                current_lr = optim_img.param_groups[0]["lr"]
                plotter.plot_and_save_loss_curve()
                if args.iter_calib > 0:
                    args.logger(
                        f"\n{get_time()} (Iter {it:3d}) "
                        f"LR: {current_lr:.6f} "
                        f"inter-loss: {calib_loss_total / args.nclass / args.iter_calib:.2f} "
                        f"inner-loss: {match_loss_total / args.nclass:.2f} "
                        f"grad-norm: {total_grad_mean / args.nclass:.7f} "
                        f"Timing Stats: {timing_stats}"
                    )
                else:
                    args.logger(
                        f"\n{get_time()} (Iter {it:3d}) "
                        f"LR: {current_lr:.6f} "
                        f"inner-loss: {match_loss_total / args.nclass:.2f} "
                        f"grad-norm: {total_grad_mean / args.nclass:.7f} "
                        f"Timing Stats: {timing_stats}"
                    )
            if (it + 1) in args.it_save:
                gather_save_visualize(self, args, iteration=it)
            scheduler.step(current_loss)
            if scheduler_sampling_net is not None:
                scheduler_sampling_net.step(current_loss)

    def evaluate(self, args, syndataloader, val_loader):
        if args.rank == 0:
            args.logger("======================Start Evaluation ======================")
        acc_results = []
        macro_f1_results = []
        macro_precision_results = []
        macro_recall_results = []
        macro_specificity_results = []
        best_global_macro_f1 = -1.0
        best_model_state = None
        all_macro_histories = []  # list of per-repeat macro history
        for i in range(args.val_repeat):
            if args.rank == 0:
                args.logger(
                    f"======================Repeat {i+1}/{args.val_repeat} Starting =================================================================="
                )
            # Set a distinct seed for each repeat to ensure different randomness per evaluation
            try:
                from utils.init_script import set_random_seeds
                base_seed = getattr(args, 'seed', 0) or 0
                per_repeat_seed = int(base_seed) + i + 1
                if per_repeat_seed > 0:
                    set_random_seeds(per_repeat_seed)
                    if args.rank == 0:
                        args.logger(f"Per-repeat seed set to {per_repeat_seed} for repeat {i+1}")
            except Exception:
                # If setting seed fails, continue without crashing
                if args.rank == 0:
                    args.logger(f"[WARN] Failed to set per-repeat seed for repeat {i+1}")
            model = define_model(
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
            (
                best_acc,
                acc,
                best_macro_f1,
                best_macro_f1_state,
                macro_history,
                best_macro_f1_cm,
                best_macro_f1_metrics,
                history,
            ) = evaluate_syn_data(
                args, model, syndataloader, val_loader, logger=args.logger
            )
            acc_results.append(best_acc)
            all_macro_histories.append(macro_history)
            # 保存每次 repeat 的训练/验证曲线与历史 JSON
            if args.rank == 0:
                try:
                    from utils.plot_utils import plot_train_val_curves
                    save_dir = getattr(args, 'save_dir', './outputs')
                    os.makedirs(save_dir, exist_ok=True)
                    png_path = os.path.join(save_dir, f'repeat_{i+1}_train_val_curves.png')
                    title_prefix = f'Repeat {i+1}'
                    # Clean history: convert None to nan for plotting
                    hist_for_plot = {k: list(v) for k, v in history.items()}
                    import math
                    for k in ['val_top1', 'val_top3', 'val_loss']:
                        hist_for_plot[k] = [float(x) if (x is not None) else math.nan for x in hist_for_plot.get(k, [])]
                    plot_train_val_curves(hist_for_plot, save_path=png_path, title_prefix=title_prefix)
                    # save JSON history
                    import json
                    json_path = os.path.join(save_dir, f'repeat_{i+1}_history.json')
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(history, f, ensure_ascii=False, indent=2)
                    args.logger(f"Saved repeat {i+1} train/val curves to {png_path} and history to {json_path}")
                except Exception as e:
                    args.logger(f"[WARN] Failed to save train/val curves for repeat {i+1}: {e}")
            # 取该 repeat 内最后一次（即 macro_history 最后一个）的宏指标做跨 repeat 统计
            if macro_history:
                last_metrics = macro_history[-1]
                macro_f1_results.append(last_metrics['macro_f1'])
                macro_precision_results.append(last_metrics['macro_precision'])
                macro_recall_results.append(last_metrics['macro_recall'])
                macro_specificity_results.append(last_metrics['macro_specificity'])
            else:
                macro_f1_results.append(0.0)
                macro_precision_results.append(0.0)
                macro_recall_results.append(0.0)
                macro_specificity_results.append(0.0)
            # 更新全局最佳 macro F1 模型
            if best_macro_f1_state is not None and best_macro_f1 > best_global_macro_f1:
                best_global_macro_f1 = best_macro_f1
                best_model_state = best_macro_f1_state
                best_model_cm = best_macro_f1_cm
                best_model_metrics = best_macro_f1_metrics
            if args.rank == 0:
                args.logger(
                    f"Repeat {i+1}/{args.val_repeat} => BestAcc(top1): {best_acc:.3f}  LastAcc: {acc:.3f}  BestMacroF1: {best_macro_f1:.4f}\n"
                )

        # 统计聚合
        acc_mean, acc_std = np.mean(acc_results), np.std(acc_results)
        macro_f1_mean, macro_f1_std = np.mean(macro_f1_results), np.std(macro_f1_results)
        macro_precision_mean = np.mean(macro_precision_results)
        macro_recall_mean = np.mean(macro_recall_results)
        macro_specificity_mean = np.mean(macro_specificity_results)
        if args.rank == 0:
            args.logger("=" * 60)
            args.logger("Evaluation Summary Across Repeats:")
            args.logger(
                f"Accuracy Mean {acc_mean:.3f} Std {acc_std:.3f} | MacroF1 Mean {macro_f1_mean:.4f} Std {macro_f1_std:.4f}"
            )
            args.logger(
                f"Macro Precision {macro_precision_mean:.4f} Recall {macro_recall_mean:.4f} Specificity {macro_specificity_mean:.4f}"
            )
            args.logger(f"Per-Repeat MacroF1: {[f'{x:.4f}' for x in macro_f1_results]}")
            # 保存最佳 Macro F1 模型权重
            if best_model_state is not None:
                save_dir = getattr(args, 'save_dir', './outputs')
                os.makedirs(save_dir, exist_ok=True)
                # 1) 保存权重
                best_path = os.path.join(save_dir, 'best_macro_f1_model.pt')
                try:
                    torch.save(best_model_state, best_path)
                    args.logger(f"Saved best macro F1 model weights to {best_path} (F1={best_global_macro_f1:.4f})")
                except Exception as e:
                    args.logger(f"[WARN] Failed to save best macro F1 model weights: {e}")
                # 2) 保存指标 JSON
                import json
                metrics_path = os.path.join(save_dir, 'best_macro_f1_metrics.json')
                try:
                    with open(metrics_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            'macro_f1': best_global_macro_f1,
                            'metrics': best_model_metrics
                        }, f, ensure_ascii=False, indent=2)
                    args.logger(f"Saved best macro F1 metrics JSON to {metrics_path}")
                except Exception as e:
                    args.logger(f"[WARN] Failed to save metrics JSON: {e}")
                # 3) 绘制并保存混淆矩阵 PNG (归一化显示)
                try:
                    from utils.plot_utils import plot_confusion_matrix
                    if best_model_cm is not None:
                        cm_png = os.path.join(save_dir, 'best_macro_f1_confusion_matrix.png')
                        plot_confusion_matrix(best_model_cm, normalize=True, save_path=cm_png, title='Best Macro F1 Confusion Matrix')
                        args.logger(f"Saved best macro F1 confusion matrix figure to {cm_png}")
                except Exception as e:
                    args.logger(f"[WARN] Failed to save confusion matrix figure: {e}")
            args.logger("=" * 60)

    def continue_learning(self, args, syndataloader, val_loader):
        if args.rank == 0:
            args.logger("Start Continue Learning ......... :D ")
        mean_result_list = []
        std_result_list = []
        results = []
        all_best_acc = 0
        step_classes = len(self.nclass_list) // args.steps

        all_classes = list(range(self.nclass))
        for current_step in range(1, args.step + 1):
            classes_seen = random.sample(all_classes, current_step * step_classes)
            def get_loader_step(classes_seen, val_loader):
                val_data, val_targets = [], []

                for data, target in val_loader:
                    mask = torch.tensor(
                        [t.item() in classes_seen for t in target], device=target.device
                    )
                    val_data.append(data[mask])
                    val_targets.append(target[mask])

                val_data = torch.cat(val_data)
                val_targets = torch.cat(val_targets)

                val_dataset_step = TensorDataset(val_data, val_targets)
                val_loader_step = DataLoader(val_dataset_step, batch_size=128, shuffle=False)
                return val_loader_step

            val_loader_step = get_loader_step(classes_seen, val_loader)
            syndataloader = get_loader_step(classes_seen, syndataloader)
            for i in range(args.val_repeat):
                args.logger(
                    f"======================Repeat {i+1}/{args.val_repeat} Starting =================================================================="
                )
                model = define_model(
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
                best_acc, acc = evaluate_syn_data(
                    args, model, syndataloader, val_loader_step, logger=args.logger
                )
                if all_best_acc < best_acc:
                    all_best_acc = best_acc
                results.append(best_acc)
                if args.rank == 0:
                    args.logger(
                        f"Step {current_step},Repeat {i+1}/{args.val_repeat} => The Best Evaluation Acc: {all_best_acc:.1f} The Last Evaluation Acc :{acc:.1f} \n"
                    )
            mean_result = np.mean(results)
            std_result = np.std(results)
            mean_result_list.append(mean_result)
            std_result_list.append(std_result)
        if args.rank == 0:
            args.logger("=" * 50)
            args.logger(
                f"All result: {[f'Step {i} Acc: {x:.3f}' for i, x in enumerate(mean_result_list)]}"
            )
            args.logger("=" * 50)
