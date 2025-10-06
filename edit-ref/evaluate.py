import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.experiment_tracker import get_time
from utils.diffaug import DiffAug, ClasswiseDiffAugWrapper, get_augmentation_summary, write_aug_summary_json
from utils.utils import define_model
from utils.ddp import load_state_dict
import warnings
from utils.train_val import train_epoch, validate, train_epoch_softlabel
try:
    from utils.utils import ensure_dir  # if already exists
except Exception:
    import os as _os
    def ensure_dir(path: str):
        if path is None:
            return
        try:
            _os.makedirs(path, exist_ok=True)
        except Exception:
            pass
from utils import utils as _u  # fallback if needed
import math

# Lazy import plotting util (will add in utils)
try:
    from utils.plot_utils import plot_confusion_matrix
except Exception:
    plot_confusion_matrix = None

warnings.filterwarnings("ignore")
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
import os
import json


def SoftCrossEntropy(inputs, target, temperature=1.0, reduction="average"):
    input_log_likelihood = -F.log_softmax(inputs / temperature, dim=1)
    target_log_likelihood = F.softmax(target / temperature, dim=1)
    batch = inputs.shape[0]
    loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
    return loss


# loss_function_kl = nn.KLDivLoss(reduction="batchmean")
def evaluate_syn_data(args, model, train_loader, val_loader, logger=None):
    if args.softlabel:
        teacher_model = define_model(
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
        teacher_path = os.path.join(args.pretrain_dir, f"premodel0_trained.pth.tar")
        load_state_dict(teacher_path, teacher_model)
        train_criterion_sl = SoftCrossEntropy
    train_criterion = nn.CrossEntropyLoss().cuda()
    val_criterion = nn.CrossEntropyLoss().cuda()
    if args.eval_optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.adamw_lr)
        if logger and dist.get_rank() == 0:
            logger(f"Using AdamW optimizer with learning rate: {args.adamw_lr}")
    elif args.eval_optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum
        )
        if logger and dist.get_rank() == 0:
            logger(f"Using SGD optimizer with learning rate: {args.lr}")
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            args.evaluation_epochs // 5,
            2 * args.evaluation_epochs // 5,
            3 * args.evaluation_epochs // 5,
            4 * args.evaluation_epochs // 5,
        ],
        gamma=0.5,
    )
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[args.evaluation_epochs//2], gamma=0.1)

    best_acc1, best_acc3 = 0, 0
    acc1, acc3 = 0, 0
    best_macro_f1 = -1.0
    best_macro_f1_state = None  # store model weights (CPU) for highest macro F1
    best_macro_f1_cm = None     # store confusion matrix (numpy) at best macro F1
    best_macro_f1_metrics = None  # store metrics dict at best macro F1
    macro_history = []  # list of dicts per evaluation step
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.rank], output_device=args.rank
    )

    # Evaluation 阶段：使用 YAML 中 dsa 与 dsa_strategy 控制是否启用 DSA；
    aug = None
    if getattr(args, 'dsa', False):
        strategy = getattr(args, 'dsa_strategy', 'color_crop_cutout_flip_scale_rotate')
        base_aug = DiffAug(strategy=strategy, batch=True)
        # 可选分类别增强
        if getattr(args, 'classwise_aug', False):
            aug = ClasswiseDiffAugWrapper(base_aug, args, device=args.device)
        else:
            aug = base_aug
        if args.rank == 0 and logger:
            logger(f"Evaluation uses DSA(strategy={strategy}) classwise_aug={getattr(args,'classwise_aug',False)}")
            # 写入 JSON 增强摘要
            if hasattr(args, 'save_dir'):
                try:
                    os.makedirs(args.save_dir, exist_ok=True)
                    json_path = os.path.join(args.save_dir, 'augmentation_log.jsonl')
                    write_aug_summary_json(json_path, 'evaluation', aug, extra={'evaluation_epochs': args.evaluation_epochs})
                    logger(get_augmentation_summary(aug))
                except Exception as e:
                    logger(f"[WARN] Failed to write evaluation augmentation summary: {e}")
    else:
        if args.rank == 0 and logger:
            logger("Evaluation without DSA (dsa=false)")
            
    pbar = tqdm(range(1, args.evaluation_epochs + 1))
    # History collection per epoch
    history = {
        'epoch': [],
        'train_top1': [],
        'train_top3': [],
        'train_loss': [],
        'val_top1': [],
        'val_top3': [],
        'val_loss': [],
    }
    for epoch in range(1, args.evaluation_epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        if args.softlabel and epoch < (
            args.evaluation_epochs - args.epoch_eval_interval
        ):
            acc1_tr, acc3_tr, loss_tr = train_epoch_softlabel(
                args,
                train_loader,
                model,
                teacher_model,
                train_criterion_sl,
                optimizer,
                epoch,
                aug,
                mixup=args.mixup,
            )
        else:
            acc1_tr, acc3_tr, loss_tr = train_epoch(
                args,
                train_loader,
                model,
                train_criterion,
                optimizer,
                epoch,
                aug,
                mixup="none",
            )
        if args.rank == 0:
            pbar.set_description(
                f"[Epoch {epoch}/{args.evaluation_epochs}] (Train) Top1 {acc1_tr:.1f}  Top3 {acc3_tr:.1f} Lr {optimizer.param_groups[0]['lr']} Loss {loss_tr:.3f}"
            )
            pbar.update(1)
            if (epoch % args.epoch_print_freq == 0) and (logger is not None):
                logger(
                    "(Train) [Epoch {0}/{1}] {2} Top1 {top1:.1f}  Top3 {top3:.1f}  Loss {loss:.3f}".format(
                        epoch,
                        args.evaluation_epochs,
                        get_time(),
                        top1=acc1_tr,
                        top3=acc3_tr,
                        loss=loss_tr,
                    )
                )

        # Record train metrics for this epoch
        history['epoch'].append(epoch)
        history['train_top1'].append(float(acc1_tr))
        history['train_top3'].append(float(acc3_tr))
        history['train_loss'].append(float(loss_tr))

        if (
            epoch % args.epoch_eval_interval == 0
            or epoch == args.evaluation_epochs
            or (epoch % (args.epoch_eval_interval / 50) == 0 and args.ipc > 50)
        ):
            acc1, acc3, loss_val, cm = validate(val_loader, model, val_criterion, args)
            # fill val metrics for this epoch
            history['val_top1'].append(float(acc1))
            history['val_top3'].append(float(acc3))
            history['val_loss'].append(float(loss_val))

            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_acc3 = acc3
            if logger is not None and args.rank == 0:
                logger(
                    "-------Eval Training Epoch [{} / {}] INFO--------".format(
                        epoch, args.evaluation_epochs
                    )
                )
                logger(
                    f"Current accuracy (top-1 and 3): {acc1:.1f} {acc3:.1f}, loss: {loss_val:.3f}"
                )
                logger(
                    f"Best    accuracy (top-1 and 3): {best_acc1:.1f} {best_acc3:.1f}"
                )
                # Compute extended classification metrics from confusion matrix
                if cm.numel() > 0 and cm.shape[0] == cm.shape[1] and cm.shape[0] > 1:
                    cm_np = cm.numpy()
                    support_per_class = cm_np.sum(axis=1)  # true instances per class
                    pred_per_class = cm_np.sum(axis=0)
                    tp = np.diag(cm_np)
                    fp = pred_per_class - tp
                    fn = support_per_class - tp
                    tn = cm_np.sum() - (tp + fp + fn)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
                        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)  # sensitivity
                        specificity = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
                        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
                    macro_precision = precision.mean()
                    macro_recall = recall.mean()
                    macro_specificity = specificity.mean()
                    macro_f1 = f1.mean()
                    micro_tp = tp.sum()
                    micro_fp = fp.sum()
                    micro_fn = fn.sum()
                    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
                    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
                    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0
                    micro_tn = tn.sum()
                    micro_specificity = micro_tn / (micro_tn + micro_fp) if (micro_tn + micro_fp) > 0 else 0.0
                    logger(
                        f"Extended Metrics (Macro) | Precision {macro_precision:.4f} Recall(Sens) {macro_recall:.4f} Specificity {macro_specificity:.4f} F1 {macro_f1:.4f}"
                    )
                    logger(
                        f"Extended Metrics (Micro) | Precision {micro_precision:.4f} Recall(Sens) {micro_recall:.4f} Specificity {micro_specificity:.4f} F1 {micro_f1:.4f}"
                    )
                    # Weighted metrics (by support / class sample counts)
                    total_support = support_per_class.sum() if support_per_class.sum() > 0 else 1
                    weights = support_per_class / total_support
                    weighted_precision = float(np.sum(precision * weights))
                    weighted_recall = float(np.sum(recall * weights))
                    weighted_specificity = float(np.sum(specificity * weights))
                    weighted_f1 = float(np.sum(f1 * weights))

                    macro_history.append({
                        'epoch': epoch,
                        'macro_precision': float(macro_precision),
                        'macro_recall': float(macro_recall),
                        'macro_specificity': float(macro_specificity),
                        'macro_f1': float(macro_f1),
                        'micro_precision': float(micro_precision),
                        'micro_recall': float(micro_recall),
                        'micro_specificity': float(micro_specificity),
                        'micro_f1': float(micro_f1),
                        'weighted_precision': weighted_precision,
                        'weighted_recall': weighted_recall,
                        'weighted_specificity': weighted_specificity,
                        'weighted_f1': weighted_f1
                    })
                    # Track best macro F1 model and retain confusion matrix + metrics
                    if macro_f1 > best_macro_f1:
                        best_macro_f1 = float(macro_f1)
                        try:
                            best_macro_f1_state = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
                            best_macro_f1_cm = cm_np.copy()
                            best_macro_f1_metrics = macro_history[-1].copy()
                            # Persist best model weights immediately if path provided
                            save_dir = getattr(args, 'save_dir', './outputs')
                            ensure_dir(save_dir)
                            best_model_path = os.path.join(save_dir, 'best_macro_f1_model.pth')
                            torch.save({'state_dict': best_macro_f1_state,
                                        'epoch': epoch,
                                        'macro_f1': best_macro_f1,
                                        'metrics': best_macro_f1_metrics}, best_model_path)
                            logger(f"[Model Saved] New best macro F1 {best_macro_f1:.4f} at epoch {epoch} -> {best_model_path}")
                        except Exception as e:
                            logger(f"[WARN] Failed to copy/save best macro F1 state dict: {e}")
                    # Save confusion matrix plot when best or final epoch
                    if plot_confusion_matrix is not None and (is_best or epoch == args.evaluation_epochs):
                        save_dir = getattr(args, 'save_dir', './outputs')
                        ensure_dir(save_dir)
                        cm_path = os.path.join(save_dir, f'confusion_matrix_epoch{epoch}_best{int(is_best)}.png')
                        try:
                            class_names = getattr(args, 'class_names', None)
                            plot_confusion_matrix(
                                cm_np,
                                class_names=class_names,
                                normalize=True,
                                save_path=cm_path,
                                title=f'Confusion Matrix (Epoch {epoch})'
                            )
                            logger(f"Saved confusion matrix to {cm_path}")
                        except Exception as e:
                            logger(f"[WARN] Failed to save confusion matrix plot: {e}")
        else:
            # no validation this epoch -> append placeholders
            history['val_top1'].append(None)
            history['val_top3'].append(None)
            history['val_loss'].append(None)

        # At final evaluation step produce a classification report file (simple version) from stored metrics history
        if epoch == args.evaluation_epochs and args.rank == 0 and logger is not None:
            try:
                save_dir = getattr(args, 'save_dir', './outputs')
                ensure_dir(save_dir)
                # Build a per-class classification summary if last cm available
                if 'cm_np' in locals():
                    cm_final = cm_np  # last computed
                    per_class = []
                    for c in range(cm_final.shape[0]):
                        tp_c = cm_final[c, c]
                        support_c = cm_final[c, :].sum()
                        pred_c = cm_final[:, c].sum()
                        fp_c = pred_c - tp_c
                        fn_c = support_c - tp_c
                        tn_c = cm_final.sum() - (tp_c + fp_c + fn_c)
                        precision_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
                        recall_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
                        specificity_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0.0
                        f1_c = 2 * precision_c * recall_c / (precision_c + recall_c) if (precision_c + recall_c) > 0 else 0.0
                        per_class.append({
                            'class_index': int(c),
                            'precision': float(precision_c),
                            'recall': float(recall_c),
                            'specificity': float(specificity_c),
                            'f1': float(f1_c),
                            'support': int(support_c)
                        })
                    class_names = getattr(args, 'class_names', None)
                    if class_names and len(class_names) == len(per_class):
                        for i, item in enumerate(per_class):
                            item['class_name'] = class_names[i]
                    report = {
                        'epoch': epoch,
                        'best_macro_f1': best_macro_f1,
                        'best_macro_f1_metrics': best_macro_f1_metrics,
                        'per_class': per_class,
                        'macro_history': macro_history,
                    }
                    report_path = os.path.join(save_dir, 'classification_report.json')
                    with open(report_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, ensure_ascii=False, indent=2)
                    logger(f"Saved classification report to {report_path}")
            except Exception as e:
                logger(f"[WARN] Failed to generate classification report: {e}")

        scheduler.step()

    return (
        best_acc1,
        acc1,
        best_macro_f1,
        best_macro_f1_state,
        macro_history,
        best_macro_f1_cm,
        best_macro_f1_metrics,
        history,
    )
