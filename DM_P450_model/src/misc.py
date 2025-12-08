import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score as calc_auc,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


def run_one_epoch(
    model, data_loader, optimizer, loss_fn, writer, epoch, step=None, mode="train"
):
    """
    通用训练/验证函数
    mode: "train" 或 "val"
    """
    is_train = mode == "train"
    device = next(model.parameters()).device
    total_loss = 0
    evaluate_metrics = False
    if epoch % 10 == 0:
        evaluate_metrics = True
    if evaluate_metrics:
        all_probs, all_labels = [], []
    if is_train:
        model.train()
        grad_context = torch.enable_grad()
    else:
        model.eval()
        grad_context = torch.no_grad()
    with grad_context:
        for batch in tqdm(data_loader, desc=f"Epoch {epoch} [{mode}]", leave=False):
            (
                batched_graphs,
                batched_labels,
                batched_scores,
                batched_seqs,
                batched_substrates,
            ) = [x.to(device) for x in batch[:5]]

            node_features = {
                "ligand": batched_graphs.nodes["ligand"].data["h"],
                "protein": batched_graphs.nodes["protein"].data["h"],
            }
            predictions = model(
                (batched_graphs, node_features, batched_scores),
                (batched_seqs, batched_substrates),
            )
            # loss
            loss = loss_fn(predictions.view(-1), batched_labels.view(-1).float())
            total_loss += loss.item()
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if writer is not None:
                    writer.add_scalar(f"Loss/{mode}/batch", loss.item(), step)
                step += 1
            if evaluate_metrics:
                probs = torch.sigmoid(predictions).detach().cpu().numpy().reshape(-1)
                labels = batched_labels.detach().cpu().numpy().reshape(-1)
                all_probs.extend(probs)
                all_labels.extend(labels)

    avg_loss = total_loss / len(data_loader)
    if writer is not None:
        writer.add_scalar(f"Loss/{mode}/epoch", avg_loss, epoch)

    if evaluate_metrics:
        # 计算 ROC 曲线
        fpr_list, tpr_list, thresholds = roc_curve(all_labels, all_probs)
        # 找最大化 F1 的阈值
        f1_scores = [
            f1_score(all_labels, (all_probs > thr).astype(int)) for thr in thresholds
        ]
        best_threshold = thresholds[np.argmax(f1_scores)]
        # 用最优阈值做预测
        preds = (all_probs > best_threshold).astype(int)
        # 重新计算指标
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)
        auc = calc_auc(all_labels, all_probs)
        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        if writer is not None:
            writer.add_scalar(f"Metrics/{mode}/acc", acc, epoch)
            writer.add_scalar(f"Metrics/{mode}/f1", f1, epoch)
            writer.add_scalar(f"Metrics/{mode}/auc", auc, epoch)
            writer.add_scalar(f"Metrics/{mode}/tpr", tpr, epoch)
            writer.add_scalar(f"Metrics/{mode}/fpr", fpr, epoch)
            writer.add_scalar(f"Metrics/{mode}/threshold", best_threshold, epoch)
            writer.add_scalar(f"Confusion/{mode}/TP", tp, epoch)
            writer.add_scalar(f"Confusion/{mode}/FP", fp, epoch)
            writer.add_scalar(f"Confusion/{mode}/TN", tn, epoch)
            writer.add_scalar(f"Confusion/{mode}/FN", fn, epoch)
    if is_train:
        return avg_loss, step
    else:
        return avg_loss


def save_checkpoint(
    model,
    optimizer,
    epoch,
    step,
    model_path="data/model/checkpoint.pth",
    optim_path="data/optimizer/checkpoint.pth",
):
    # 保存模型参数
    torch.save(model.state_dict(), model_path)

    # 保存优化器状态和训练信息
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, optim_path)


def load_checkpoint(
    model,
    optimizer,
    model_path="data/model/checkpoint.pth",
    optim_path="data/optimizer/checkpoint.pth",
    device="cpu",
):
    # 加载模型参数
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False)
        )
        # 加载优化器状态
        checkpoint = torch.load(optim_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        step = checkpoint["step"]
        print(f"✅ Loaded model from {model_path}")
        print(
            f"✅ Loaded optimizer state from {optim_path} (resume from epoch {start_epoch})"
        )
        return start_epoch, step
    except FileNotFoundError:
        print("⚠️ Checkpoint files not found. Starting from scratch.")
        return 1, 1


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha: 正样本权重（类似 BCE 的 pos_weight）
            gamma: 聚焦参数，γ>1 时更关注 hard 样本
            reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # 先算概率
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)  # p_t

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BCEWithRankingFocalLoss(nn.Module):
    def __init__(
        self, pos_weight=None, margin=1.0, alpha=1.0, beta=1.0, reduction="mean"
    ):
        """
        Args:
            pos_weight: BCE 部分的正样本权重
            margin: RankingLoss 的 margin
            alpha: RankingLoss 权重
            beta: FocalLoss 权重
            reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
        self.rank_loss = nn.MarginRankingLoss(margin=margin, reduction=reduction)
        self.focal_loss = FocalLoss(alpha=1.0, gamma=3.0, reduction=reduction)
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        # BCE
        bce = self.bce_loss(logits, targets)

        # Ranking
        pos_mask = targets > 0.5
        neg_mask = targets < 0.5
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_logits = logits[pos_mask]
            neg_logits = logits[neg_mask]

            k = min(50, neg_logits.size(0))
            topk_vals, _ = torch.topk(neg_logits, k=k)
            neg_logits = topk_vals

            pos_pairs = pos_logits[:, None].expand(-1, neg_logits.size(0)).reshape(-1)
            neg_pairs = neg_logits[None, :].expand(pos_logits.size(0), -1).reshape(-1)
            y = torch.ones_like(pos_pairs, device=logits.device)

            rank = self.rank_loss(pos_pairs, neg_pairs, y)
        else:
            rank = torch.tensor(0.0, device=logits.device)

        # Focal
        focal = self.focal_loss(logits, targets)

        return bce + self.alpha * rank + self.beta * focal
