import math
import random
import sys
import time
from typing import Any, Dict, Optional, Tuple


import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn import Parameter
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from model.DataSet import TGCDataSet
from model.evaluation import eva


class TGC:

    def __init__(self, args: Any):
        self.args = args
        self.the_data = args.dataset
        self.seed = int(getattr(args, "seed", 0))
        self._set_seed_once(self.seed)

        self.file_path = f"../data/{self.the_data}/{self.the_data}.txt"
        self.feature_path = f"./pretrain/{self.the_data}_feature.emb"
        self.label_path = f"../data/{self.the_data}/node2label.txt"

        self.neg_size = int(args.neg_size)
        self.hist_len = int(args.hist_len)
        self.batch_size = int(args.batch_size)
        self.clusters = int(args.clusters)
        self.epochs = int(args.epoch)
        self.lr = float(args.learning_rate)
        self.directed = bool(args.directed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.data = TGCDataSet(
            self.file_path,
            self.neg_size,
            self.hist_len,
            self.feature_path,
            self.directed,
        )

        self.node_dim = self.data.get_node_dim()
        self.feature = self.data.get_feature().astype(np.float32)
        self.feature_dim = int(self.feature.shape[1])
        self.emb_size = self.feature_dim

        arg_emb_size = int(getattr(args, "emb_size", self.emb_size))
        if arg_emb_size != self.emb_size:
            print(
                f"[Warning] args.emb_size={arg_emb_size}, "
                f"but feature_dim={self.emb_size}. Using feature_dim={self.emb_size}."
            )

        self.labels = self.read_label(expected_nodes=self.node_dim)

        self.node_emb = Parameter(torch.from_numpy(self.feature).to(self.device))
        self.delta = Parameter(torch.ones(self.node_dim, device=self.device))

        self.cluster_layer = Parameter(
            torch.empty(self.clusters, self.emb_size, device=self.device)
        )
        with torch.no_grad():
            self.cluster_layer.copy_(self.init_global_cluster_centers(self.feature))

        self.use_original_kl_loss = bool(int(getattr(args, "use_original_kl_loss", 1)))
        self.legacy_loss_reduction = str(
            getattr(args, "legacy_loss_reduction", "sum")
        ).lower()
        if self.legacy_loss_reduction not in {"sum", "mean"}:
            raise ValueError("legacy_loss_reduction must be 'sum' or 'mean'.")

        self.v = float(getattr(args, "v", 1.0))

        self.use_membership_entropy_kl_weight = bool(
            int(getattr(args, "use_membership_entropy_kl_weight", 1))
        )
        self.membership_kl_beta = float(getattr(args, "membership_kl_beta", 0.20))
        self.membership_kl_weight_min = float(
            getattr(args, "membership_kl_weight_min", 0.80)
        )
        self.membership_kl_weight_max = float(
            getattr(args, "membership_kl_weight_max", 1.20)
        )
        self.membership_kl_detach = bool(int(getattr(args, "membership_kl_detach", 1)))
        self.membership_kl_mean_normalize = bool(
            int(getattr(args, "membership_kl_mean_normalize", 1))
        )
        self.membership_kl_warmup_epoch = int(
            getattr(args, "membership_kl_warmup_epoch", 20)
        )

        if self.membership_kl_beta < 0.0:
            raise ValueError("membership_kl_beta must be non-negative.")
        if self.membership_kl_weight_min <= 0.0:
            raise ValueError("membership_kl_weight_min must be positive.")
        if self.membership_kl_weight_max < self.membership_kl_weight_min:
            raise ValueError(
                "membership_kl_weight_max must be >= membership_kl_weight_min."
            )
        if self.membership_kl_warmup_epoch < 0:
            raise ValueError("membership_kl_warmup_epoch must be non-negative.")

        self.beta1 = float(getattr(args, "beta1", 1.0))
        self.beta2 = float(getattr(args, "beta2", 1.0))
        self.fuzzy_delta = float(getattr(args, "fuzzy_delta", 0.1))
        self.eta0 = float(getattr(args, "eta0", 1.0))
        self.eta1 = float(getattr(args, "eta1", 1.0))
        self.normalize_fuzzy_time = bool(int(getattr(args, "normalize_fuzzy_time", 1)))
        self.fuzzy_logit_clamp = float(getattr(args, "fuzzy_logit_clamp", 60.0))

        self.use_fuzzy_weighting = bool(int(getattr(args, "use_fuzzy_weighting", 1)))
        self.fuzzy_weight_strength = float(getattr(args, "fuzzy_weight_strength", 0.02))
        self.warmup_epochs = int(getattr(args, "warmup_epochs", 20))
        self.fuzzy_ramp_epochs = int(getattr(args, "fuzzy_ramp_epochs", 30))
        self.detach_fuzzy_weight = bool(int(getattr(args, "detach_fuzzy_weight", 1)))
        self.normalize_fuzzy_weight = bool(
            int(getattr(args, "normalize_fuzzy_weight", 1))
        )
        self.fuzzy_weight_centered = bool(
            int(getattr(args, "fuzzy_weight_centered", 1))
        )
        self.fuzzy_weight_clip = float(getattr(args, "fuzzy_weight_clip", 2.0))
        self.fuzzy_weight_min = float(getattr(args, "fuzzy_weight_min", 0.1))
        self.fuzzy_weight_max = float(getattr(args, "fuzzy_weight_max", 10.0))

        if self.fuzzy_weight_min <= 0.0:
            raise ValueError("fuzzy_weight_min must be positive.")
        if self.fuzzy_weight_max < self.fuzzy_weight_min:
            raise ValueError("fuzzy_weight_max must be >= fuzzy_weight_min.")

        self.use_reliability_gated_weight = bool(
            int(getattr(args, "use_reliability_gated_weight", 1))
        )
        self.reliability_entropy_power = float(
            getattr(args, "reliability_entropy_power", 1.0)
        )
        self.reliability_min = float(getattr(args, "reliability_min", 0.25))
        self.reliability_max = float(getattr(args, "reliability_max", 1.0))
        if not (0.0 <= self.reliability_min <= self.reliability_max <= 1.0):
            raise ValueError("Require 0 <= reliability_min <= reliability_max <= 1.")
        if self.reliability_entropy_power <= 0.0:
            raise ValueError("reliability_entropy_power must be positive.")

        self.use_fuzzy_hawkes_triggering = bool(
            int(getattr(args, "use_fuzzy_hawkes_triggering", 1))
        )
        self.fuzzy_hawkes_triggering_warmup_epoch = int(
            getattr(args, "fuzzy_hawkes_triggering_warmup_epoch", 20)
        )
        self.fuzzy_hawkes_triggering_strength = float(
            getattr(args, "fuzzy_hawkes_triggering_strength", 0.02)
        )
        self.fuzzy_hawkes_triggering_tail_clip = float(
            getattr(args, "fuzzy_hawkes_triggering_tail_clip", 1.0)
        )
        self.fuzzy_hawkes_triggering_min = float(
            getattr(args, "fuzzy_hawkes_triggering_min", 0.98)
        )
        self.fuzzy_hawkes_triggering_max = float(
            getattr(args, "fuzzy_hawkes_triggering_max", 1.00)
        )
        self.fuzzy_hawkes_triggering_temp = float(
            getattr(args, "fuzzy_hawkes_triggering_temp", 1.0)
        )
        self.fuzzy_hawkes_triggering_comp_weight = float(
            getattr(args, "fuzzy_hawkes_triggering_comp_weight", 1.0)
        )
        self.fuzzy_hawkes_triggering_att_weight = float(
            getattr(args, "fuzzy_hawkes_triggering_att_weight", 0.5)
        )
        self.fuzzy_hawkes_triggering_neg_reduce = str(
            getattr(args, "fuzzy_hawkes_triggering_neg_reduce", "mean")
        ).lower()
        self.fuzzy_hawkes_triggering_detach = bool(
            int(getattr(args, "fuzzy_hawkes_triggering_detach", 1))
        )

        if self.fuzzy_hawkes_triggering_warmup_epoch < 0:
            raise ValueError(
                "fuzzy_hawkes_triggering_warmup_epoch must be non-negative."
            )
        if self.fuzzy_hawkes_triggering_strength < 0.0:
            raise ValueError("fuzzy_hawkes_triggering_strength must be non-negative.")
        if self.fuzzy_hawkes_triggering_tail_clip <= 0.0:
            raise ValueError("fuzzy_hawkes_triggering_tail_clip must be positive.")
        if self.fuzzy_hawkes_triggering_min <= 0.0:
            raise ValueError("fuzzy_hawkes_triggering_min must be positive.")
        if self.fuzzy_hawkes_triggering_max < self.fuzzy_hawkes_triggering_min:
            raise ValueError(
                "fuzzy_hawkes_triggering_max must be >= fuzzy_hawkes_triggering_min."
            )
        if self.fuzzy_hawkes_triggering_temp <= 0.0:
            raise ValueError("fuzzy_hawkes_triggering_temp must be positive.")
        if (
            self.fuzzy_hawkes_triggering_comp_weight < 0.0
            or self.fuzzy_hawkes_triggering_att_weight < 0.0
            or (
                self.fuzzy_hawkes_triggering_comp_weight
                + self.fuzzy_hawkes_triggering_att_weight
            )
            <= 0.0
        ):
            raise ValueError(
                "fuzzy_hawkes_triggering comp/att weights must be non-negative "
                "and not both zero."
            )
        if self.fuzzy_hawkes_triggering_neg_reduce not in {"mean", "max", "logsumexp"}:
            raise ValueError(
                "fuzzy_hawkes_triggering_neg_reduce must be one of: "
                "mean, max, logsumexp."
            )

        self.use_lr_decay = bool(int(getattr(args, "use_lr_decay", 1)))
        self.lr_decay_epoch = int(getattr(args, "lr_decay_epoch", 70))
        self.lr_decay_gamma = float(getattr(args, "lr_decay_gamma", 0.3))
        self.lr_decay_applied = False
        if self.lr_decay_epoch < 0:
            raise ValueError("lr_decay_epoch must be non-negative.")
        if not (0.0 < self.lr_decay_gamma <= 1.0):
            raise ValueError("lr_decay_gamma must be in (0, 1].")

        self.eps = float(getattr(args, "eps", 1e-8))
        self.time_eps = float(getattr(args, "time_eps", 1e-6))

        self.current_epoch = 0
        self.weight_decay = float(getattr(args, "weight_decay", 0.0))
        opt_name = str(getattr(args, "optimizer", "adam")).lower()

        trainable_params = [self.node_emb, self.delta, self.cluster_layer]
        if opt_name == "sgd":
            self.opt = SGD(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.opt = Adam(
                trainable_params, lr=self.lr, weight_decay=self.weight_decay
            )

        self.best_records = {
            "nmi": self._empty_best_record(),
            "ari": self._empty_best_record(),
            "acc": self._empty_best_record(),
            "f1": self._empty_best_record(),
            "avg": self._empty_best_record(),
        }
        self.best_nmi = 0.0
        self.best_ari = 0.0
        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.best_avg = 0.0
        self.running_loss = 0.0

    @staticmethod
    def _set_seed_once(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _empty_best_record() -> Dict[str, float]:
        return {
            "epoch": -1,
            "nmi": -float("inf"),
            "ari": -float("inf"),
            "acc": -float("inf"),
            "f1": -float("inf"),
            "avg": -float("inf"),
        }

    def current_lr(self) -> float:
        return float(self.opt.param_groups[0]["lr"])

    def apply_lr_schedule(self, epoch: int) -> Dict[str, float]:
        applied_now = 0.0
        if (
            self.use_lr_decay
            and not self.lr_decay_applied
            and epoch >= self.lr_decay_epoch
        ):
            for group in self.opt.param_groups:
                group["lr"] *= self.lr_decay_gamma
            self.lr_decay_applied = True
            applied_now = 1.0

        return {
            "current_lr": self.current_lr(),
            "lr_decay_applied": float(self.lr_decay_applied),
            "lr_decay_applied_now": applied_now,
        }

    def init_global_cluster_centers(self, x_init: np.ndarray) -> torch.Tensor:
        kmeans = KMeans(n_clusters=self.clusters, n_init=20, random_state=self.seed)
        kmeans.fit_predict(x_init)
        return torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=self.device
        )

    def read_label(self, expected_nodes: Optional[int] = None):
        n2l: Dict[int, int] = {}
        with open(self.label_path, "r") as reader:
            for line in reader:
                parts = line.strip().split()
                if len(parts) >= 2:
                    n2l[int(parts[0])] = int(parts[1])

        if expected_nodes is None:
            return [n2l[k] for k in sorted(n2l.keys())]

        labels = []
        missing = []
        for node_id in range(expected_nodes):
            if node_id in n2l:
                labels.append(n2l[node_id])
            else:
                labels.append(-1)
                missing.append(node_id)

        if missing:
            print(
                f"[Warning] label file missing {len(missing)} nodes. "
                f"Example missing ids: {missing[:10]}"
            )
        return labels

    def _update_best_metrics(
        self,
        epoch: int,
        acc: float,
        nmi: float,
        ari: float,
        f1: float,
    ) -> None:
        avg = (float(nmi) + float(ari) + float(acc) + float(f1)) / 4.0
        current = {
            "nmi": float(nmi),
            "ari": float(ari),
            "acc": float(acc),
            "f1": float(f1),
            "avg": avg,
        }
        record = {
            "epoch": int(epoch),
            "nmi": float(nmi),
            "ari": float(ari),
            "acc": float(acc),
            "f1": float(f1),
            "avg": avg,
        }
        for metric_name, metric_value in current.items():
            if metric_value > self.best_records[metric_name][metric_name]:
                self.best_records[metric_name] = dict(record)

        primary = self.best_records["avg"]
        self.best_nmi = primary["nmi"]
        self.best_ari = primary["ari"]
        self.best_acc = primary["acc"]
        self.best_f1 = primary["f1"]
        self.best_avg = primary["avg"]

    @staticmethod
    def _format_best_record(metric_name: str, record: Dict[str, float]) -> str:
        label = "Average" if metric_name == "avg" else metric_name.upper()
        if int(record["epoch"]) < 0:
            return f"Best by {label}: NA"
        suffix = f" AVG({record['avg']:.4f})" if metric_name == "avg" else ""
        return (
            f"Best by {label}: epoch={int(record['epoch'])} "
            f"NMI({record['nmi']:.4f}) ARI({record['ari']:.4f}) "
            f"ACC({record['acc']:.4f}) F1({record['f1']:.4f}){suffix}"
        )

    def get_fuzzy_weight_strength(self) -> float:
        if not self.use_fuzzy_weighting or self.current_epoch < self.warmup_epochs:
            return 0.0
        if self.fuzzy_ramp_epochs <= 0:
            return self.fuzzy_weight_strength
        ramp_step = self.current_epoch - self.warmup_epochs + 1
        ramp_ratio = min(1.0, max(0.0, ramp_step / self.fuzzy_ramp_epochs))
        return ramp_ratio * self.fuzzy_weight_strength

    def _student_t_distribution_original(self, z_batch: torch.Tensor) -> torch.Tensor:
        dist2 = torch.cdist(z_batch, self.cluster_layer, p=2).pow(2)
        q = (1.0 + dist2 / self.v).pow(-(self.v + 1.0) / 2.0)
        return q / (q.sum(dim=1, keepdim=True) + self.eps)

    def target_distribution_original(self, q: torch.Tensor) -> torch.Tensor:
        f = q.sum(dim=0)
        weight = q.pow(2) / (f.unsqueeze(0) + self.eps)
        return weight / (weight.sum(dim=1, keepdim=True) + self.eps)

    def compute_membership_entropy_kl_weights(
        self,
        q: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        if (
            not self.use_membership_entropy_kl_weight
            or self.current_epoch < self.membership_kl_warmup_epoch
            or self.membership_kl_beta <= 0.0
        ):
            return None, {
                "membership_kl_active": 0.0,
                "membership_kl_weight_mean": 1.0,
                "membership_kl_weight_std": 0.0,
                "membership_kl_weight_min": 1.0,
                "membership_kl_weight_max": 1.0,
                "membership_certainty_mean": 0.0,
                "membership_certainty_std": 0.0,
            }

        q_for_weight = q.detach() if self.membership_kl_detach else q

        entropy = -(q_for_weight * torch.log(q_for_weight + self.eps)).sum(dim=1)
        norm_entropy = entropy / (np.log(self.clusters) + self.eps)
        norm_entropy = norm_entropy.clamp(min=0.0, max=1.0)
        certainty = 1.0 - norm_entropy

        signal = certainty - certainty.mean()
        weight = 1.0 + self.membership_kl_beta * signal
        weight = weight.clamp(
            min=self.membership_kl_weight_min,
            max=self.membership_kl_weight_max,
        )

        if self.membership_kl_mean_normalize:
            weight = weight / (weight.mean() + self.eps)

        if self.membership_kl_detach:
            weight = weight.detach()

        with torch.no_grad():
            stats = {
                "membership_kl_active": 1.0,
                "membership_kl_weight_mean": float(weight.detach().mean().cpu()),
                "membership_kl_weight_std": float(
                    weight.detach().std(unbiased=False).cpu()
                ),
                "membership_kl_weight_min": float(weight.detach().min().cpu()),
                "membership_kl_weight_max": float(weight.detach().max().cpu()),
                "membership_certainty_mean": float(certainty.detach().mean().cpu()),
                "membership_certainty_std": float(
                    certainty.detach().std(unbiased=False).cpu()
                ),
            }

        return weight, stats

    def compute_original_kl_loss(
        self,
        z_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self.use_original_kl_loss:
            return torch.tensor(0.0, device=self.device), {
                "original_kl_loss": 0.0,
                "original_kl_base_loss": 0.0,
                "membership_kl_active": 0.0,
                "membership_kl_weight_mean": 1.0,
                "membership_kl_weight_std": 0.0,
                "membership_kl_weight_min": 1.0,
                "membership_kl_weight_max": 1.0,
                "membership_certainty_mean": 0.0,
                "membership_certainty_std": 0.0,
            }

        q = self._student_t_distribution_original(z_batch)
        with torch.no_grad():
            p = self.target_distribution_original(q.detach())

        kl_per_node = F.kl_div(
            torch.log(q + self.eps),
            p,
            reduction="none",
        ).sum(dim=1)
        base_kl_loss = kl_per_node.mean()

        kl_weight, kl_weight_stats = self.compute_membership_entropy_kl_weights(q)
        if kl_weight is None:
            kl_loss = base_kl_loss
        else:
            kl_loss = (kl_weight.view_as(kl_per_node) * kl_per_node).mean()

        stats = {
            "original_kl_loss": float(kl_loss.detach().cpu()),
            "original_kl_base_loss": float(base_kl_loss.detach().cpu()),
        }
        stats.update(kl_weight_stats)

        return kl_loss, stats

    @staticmethod
    def compute_base_score(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        return -torch.sum((z_a - z_b) ** 2, dim=-1)

    def _masked_softmax(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        mask_bool = mask.bool()
        masked_logits = logits.masked_fill(~mask_bool, -1e30)
        weights = torch.softmax(masked_logits, dim=dim)
        weights = weights * mask_bool.to(weights.dtype)
        weight_sum = weights.sum(dim=dim, keepdim=True)
        return torch.where(
            weight_sum > 0,
            weights / (weight_sum + self.eps),
            torch.zeros_like(weights),
        )

    def _compute_fuzzy_time_gap(
        self,
        t_cur: torch.Tensor,
        t_hist: torch.Tensor,
        h_time_mask: torch.Tensor,
    ) -> torch.Tensor:
        time_gap = torch.clamp(t_cur.unsqueeze(1) - t_hist, min=0.0)
        time_gap = time_gap * h_time_mask
        if self.normalize_fuzzy_time:
            max_gap = time_gap.max(dim=1, keepdim=True).values
            time_gap = time_gap / (max_gap + self.time_eps)
            time_gap = time_gap * h_time_mask
        return time_gap

    def compute_fuzzy_history_attention(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        z_hist: torch.Tensor,
        t_cur: torch.Tensor,
        t_hist: torch.Tensor,
        h_time_mask: torch.Tensor,
    ) -> torch.Tensor:
        score_hi = self.compute_base_score(z_hist, z_i.unsqueeze(1))
        time_gap = self._compute_fuzzy_time_gap(t_cur, t_hist, h_time_mask)

        if z_j.dim() == 2:
            score_hj = self.compute_base_score(z_hist, z_j.unsqueeze(1))
            logits = (
                self.beta1 * score_hi
                + self.beta2 * score_hj
                - self.fuzzy_delta * time_gap
            )
            return self._masked_softmax(logits, h_time_mask > 0, dim=1)

        if z_j.dim() == 3:
            neg_size = z_j.size(1)
            score_hi_exp = score_hi.unsqueeze(1).expand(-1, neg_size, -1)
            score_hj = self.compute_base_score(z_hist.unsqueeze(1), z_j.unsqueeze(2))
            time_gap_exp = time_gap.unsqueeze(1).expand(-1, neg_size, -1)
            mask_exp = (h_time_mask > 0).unsqueeze(1).expand(-1, neg_size, -1)
            logits = (
                self.beta1 * score_hi_exp
                + self.beta2 * score_hj
                - self.fuzzy_delta * time_gap_exp
            )
            return self._masked_softmax(logits, mask_exp, dim=2)

        raise ValueError(f"Unsupported z_j dim: {z_j.dim()}")

    def compute_fuzzy_temporal_relation(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        z_hist: torch.Tensor,
        t_cur: torch.Tensor,
        t_hist: torch.Tensor,
        h_time_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = self.compute_fuzzy_history_attention(
            z_i, z_j, z_hist, t_cur, t_hist, h_time_mask
        )

        if z_j.dim() == 2:
            base_ij = self.compute_base_score(z_i, z_j)
            base_hj = self.compute_base_score(z_hist, z_j.unsqueeze(1))
            hist_term = torch.sum(alpha * base_hj, dim=1)
            logits = self.eta0 * base_ij + self.eta1 * hist_term
            logits = logits.clamp(
                min=-self.fuzzy_logit_clamp, max=self.fuzzy_logit_clamp
            )
            return torch.sigmoid(logits), alpha

        if z_j.dim() == 3:
            base_ij = self.compute_base_score(z_i.unsqueeze(1), z_j)
            base_hj = self.compute_base_score(z_hist.unsqueeze(1), z_j.unsqueeze(2))
            hist_term = torch.sum(alpha * base_hj, dim=2)
            logits = self.eta0 * base_ij + self.eta1 * hist_term
            logits = logits.clamp(
                min=-self.fuzzy_logit_clamp, max=self.fuzzy_logit_clamp
            )
            return torch.sigmoid(logits), alpha

        raise ValueError(f"Unsupported z_j dim: {z_j.dim()}")

    def compute_history_reliability(
        self,
        alpha_pos: Optional[torch.Tensor],
        h_time_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        if alpha_pos is None:
            return None, {
                "rel_gate_mean": 1.0,
                "rel_gate_std": 0.0,
                "rel_gate_min": 1.0,
                "rel_gate_max": 1.0,
            }

        with torch.no_grad():
            valid_count = h_time_mask.sum(dim=1)
            has_history = valid_count > 0
            entropy = -(
                alpha_pos.detach() * torch.log(alpha_pos.detach() + self.eps)
            ).sum(dim=1)
            max_entropy = torch.log(valid_count.clamp(min=1.0)) + self.eps
            norm_entropy = (entropy / max_entropy).clamp(min=0.0, max=1.0)
            reliability = (1.0 - norm_entropy).pow(self.reliability_entropy_power)
            reliability = torch.where(
                has_history, reliability, torch.zeros_like(reliability)
            )
            reliability = reliability.clamp(
                min=self.reliability_min, max=self.reliability_max
            )
            return reliability, {
                "rel_gate_mean": float(reliability.mean().cpu()),
                "rel_gate_std": float(reliability.std(unbiased=False).cpu()),
                "rel_gate_min": float(reliability.min().cpu()),
                "rel_gate_max": float(reliability.max().cpu()),
            }

    def _standardize_signal(self, signal: torch.Tensor) -> torch.Tensor:
        if self.fuzzy_weight_centered:
            signal = signal - signal.mean()
        if self.normalize_fuzzy_weight:
            signal = signal / (signal.std(unbiased=False) + self.eps)
        if self.fuzzy_weight_clip > 0.0:
            signal = signal.clamp(
                min=-self.fuzzy_weight_clip, max=self.fuzzy_weight_clip
            )
        return signal

    def compute_fuzzy_adaptive_weights(
        self,
        r_pos: Optional[torch.Tensor],
        current_strength: float,
        reliability: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        inactive_stats = {
            "fuzzy_weight_active": 0.0,
            "pos_weight_mean": 1.0,
            "pos_weight_std": 0.0,
            "pos_weight_min": 1.0,
            "pos_weight_max": 1.0,
        }
        if not self.use_fuzzy_weighting or r_pos is None or current_strength <= 0.0:
            return None, inactive_stats

        pos_signal = r_pos
        if reliability is not None and self.use_reliability_gated_weight:
            pos_signal = pos_signal * reliability.view_as(pos_signal)

        if self.detach_fuzzy_weight:
            pos_signal = pos_signal.detach()

        pos_weight = 1.0 + current_strength * self._standardize_signal(pos_signal)
        pos_weight = pos_weight.clamp(
            min=self.fuzzy_weight_min, max=self.fuzzy_weight_max
        )

        with torch.no_grad():
            stats = {
                "fuzzy_weight_active": 1.0,
                "pos_weight_mean": float(pos_weight.detach().mean().cpu()),
                "pos_weight_std": float(pos_weight.detach().std(unbiased=False).cpu()),
                "pos_weight_min": float(pos_weight.detach().min().cpu()),
                "pos_weight_max": float(pos_weight.detach().max().cpu()),
            }

        return pos_weight, stats

    def _time_weight(
        self,
        s_nodes: torch.Tensor,
        t_times: torch.Tensor,
        h_times: torch.Tensor,
        h_time_mask: torch.Tensor,
    ) -> torch.Tensor:
        delta_raw = self.delta.index_select(0, s_nodes.view(-1)).unsqueeze(1)
        delta_pos = F.softplus(delta_raw) + self.time_eps
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)
        d_valid = d_time * h_time_mask
        max_valid = d_valid.max(dim=1, keepdim=True).values + self.time_eps
        d_norm = d_time / max_valid
        return torch.exp(-delta_pos * d_norm) * h_time_mask

    @staticmethod
    def _masked_min(
        values: torch.Tensor, mask: torch.Tensor, dim: int = 1
    ) -> torch.Tensor:
        masked = values.masked_fill(~mask.bool(), float("inf"))
        min_val = masked.min(dim=dim, keepdim=True).values
        return torch.where(torch.isfinite(min_val), min_val, torch.zeros_like(min_val))

    @staticmethod
    def _masked_max(
        values: torch.Tensor, mask: torch.Tensor, dim: int = 1
    ) -> torch.Tensor:
        masked = values.masked_fill(~mask.bool(), -float("inf"))
        max_val = masked.max(dim=dim, keepdim=True).values
        return torch.where(torch.isfinite(max_val), max_val, torch.zeros_like(max_val))

    def _masked_minmax01(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask_bool = mask.bool()
        v_min = self._masked_min(values, mask_bool, dim=1)
        v_max = self._masked_max(values, mask_bool, dim=1)
        spread = v_max - v_min
        scaled = (values - v_min) / (spread + self.eps)
        scaled = torch.where(spread > self.eps, scaled, torch.ones_like(scaled))
        return torch.where(mask_bool, scaled.clamp(0.0, 1.0), torch.zeros_like(scaled))

    def _reduce_negative_history_compatibility(
        self, n_alpha: torch.Tensor
    ) -> torch.Tensor:
        if self.fuzzy_hawkes_triggering_neg_reduce == "mean":
            return n_alpha.mean(dim=2)
        if self.fuzzy_hawkes_triggering_neg_reduce == "max":
            return n_alpha.max(dim=2).values

        return torch.logsumexp(n_alpha, dim=2) - math.log(max(1, n_alpha.size(2)))

    def compute_fuzzy_hawkes_triggering_weights(
        self,
        p_alpha: torch.Tensor,
        n_alpha: torch.Tensor,
        att: torch.Tensor,
        h_time_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        inactive = torch.ones_like(p_alpha)
        inactive_stats = {
            "hawkes_trigger_active": 0.0,
            "hawkes_trigger_weight_mean": 1.0,
            "hawkes_trigger_weight_std": 0.0,
            "hawkes_trigger_weight_min": 1.0,
            "hawkes_trigger_weight_max": 1.0,
            "hawkes_trigger_rel_mean": 0.0,
            "hawkes_trigger_rel_std": 0.0,
            "hawkes_trigger_risk_mean": 0.0,
            "hawkes_trigger_risk_std": 0.0,
            "hawkes_trigger_tail_mean": 0.0,
            "hawkes_trigger_tail_std": 0.0,
            "hawkes_trigger_tail_ratio": 0.0,
            "hawkes_trigger_utility_mean": 0.0,
            "hawkes_trigger_utility_std": 0.0,
            "hawkes_trigger_pos_comp_mean": 0.0,
            "hawkes_trigger_neg_comp_mean": 0.0,
            "hawkes_trigger_valid_ratio": 0.0,
        }
        if (
            not self.use_fuzzy_hawkes_triggering
            or self.current_epoch < self.fuzzy_hawkes_triggering_warmup_epoch
            or self.fuzzy_hawkes_triggering_strength <= 0.0
        ):
            return inactive, inactive_stats

        mask_bool = h_time_mask > 0
        if not mask_bool.any():
            return inactive, inactive_stats

        pos_comp = p_alpha.detach() if self.fuzzy_hawkes_triggering_detach else p_alpha
        neg_comp_full = (
            n_alpha.detach() if self.fuzzy_hawkes_triggering_detach else n_alpha
        )
        att_signal = att.detach() if self.fuzzy_hawkes_triggering_detach else att
        neg_comp = self._reduce_negative_history_compatibility(neg_comp_full)

        utility = pos_comp - neg_comp
        valid_float = mask_bool.float()
        valid_count = valid_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        util_row_mean = (utility * valid_float).sum(dim=1, keepdim=True) / valid_count
        util_row_var = (((utility - util_row_mean) ** 2) * valid_float).sum(
            dim=1, keepdim=True
        ) / valid_count
        util_row_std = torch.sqrt(util_row_var + self.eps)
        util_z = (utility - util_row_mean) / (util_row_std + self.eps)
        disc_rel = torch.sigmoid(self.fuzzy_hawkes_triggering_temp * util_z)
        disc_rel = torch.where(
            mask_bool, disc_rel.clamp(0.0, 1.0), torch.ones_like(disc_rel)
        )

        att01 = self._masked_minmax01(att_signal, mask_bool)

        denom = (
            self.fuzzy_hawkes_triggering_comp_weight
            + self.fuzzy_hawkes_triggering_att_weight
        )
        trigger_rel = (
            self.fuzzy_hawkes_triggering_comp_weight * disc_rel
            + self.fuzzy_hawkes_triggering_att_weight * att01
        ) / (denom + self.eps)
        trigger_rel = torch.where(
            mask_bool, trigger_rel.clamp(0.0, 1.0), torch.ones_like(trigger_rel)
        )

        trigger_risk = torch.where(
            mask_bool, 1.0 - trigger_rel, torch.zeros_like(trigger_rel)
        )

        row_mean = (trigger_risk * valid_float).sum(dim=1, keepdim=True) / valid_count
        row_var = (((trigger_risk - row_mean) ** 2) * valid_float).sum(
            dim=1, keepdim=True
        ) / valid_count
        row_std = torch.sqrt(row_var + self.eps)
        standardized_tail = (trigger_risk - row_mean) / (row_std + self.eps)
        risk_tail = torch.clamp(
            standardized_tail, min=0.0, max=self.fuzzy_hawkes_triggering_tail_clip
        )
        risk_tail = torch.where(mask_bool, risk_tail, torch.zeros_like(risk_tail))

        weight = 1.0 - self.fuzzy_hawkes_triggering_strength * risk_tail
        weight = torch.where(mask_bool, weight, torch.ones_like(weight))
        weight = weight.clamp(
            min=self.fuzzy_hawkes_triggering_min,
            max=self.fuzzy_hawkes_triggering_max,
        )
        if self.fuzzy_hawkes_triggering_detach:
            weight = weight.detach()

        with torch.no_grad():
            valid_weight = weight.detach()[mask_bool]
            valid_rel = trigger_rel.detach()[mask_bool]
            valid_risk = trigger_risk.detach()[mask_bool]
            valid_tail = risk_tail.detach()[mask_bool]
            valid_utility = utility.detach()[mask_bool]
            valid_pos_comp = pos_comp.detach()[mask_bool]
            valid_neg_comp = neg_comp.detach()[mask_bool]
            stats = {
                "hawkes_trigger_active": 1.0,
                "hawkes_trigger_weight_mean": float(valid_weight.mean().cpu()),
                "hawkes_trigger_weight_std": float(
                    valid_weight.std(unbiased=False).cpu()
                ),
                "hawkes_trigger_weight_min": float(valid_weight.min().cpu()),
                "hawkes_trigger_weight_max": float(valid_weight.max().cpu()),
                "hawkes_trigger_rel_mean": float(valid_rel.mean().cpu()),
                "hawkes_trigger_rel_std": float(valid_rel.std(unbiased=False).cpu()),
                "hawkes_trigger_risk_mean": float(valid_risk.mean().cpu()),
                "hawkes_trigger_risk_std": float(valid_risk.std(unbiased=False).cpu()),
                "hawkes_trigger_tail_mean": float(valid_tail.mean().cpu()),
                "hawkes_trigger_tail_std": float(valid_tail.std(unbiased=False).cpu()),
                "hawkes_trigger_tail_ratio": float(
                    (valid_tail > 0.0).float().mean().cpu()
                ),
                "hawkes_trigger_utility_mean": float(valid_utility.mean().cpu()),
                "hawkes_trigger_utility_std": float(
                    valid_utility.std(unbiased=False).cpu()
                ),
                "hawkes_trigger_pos_comp_mean": float(valid_pos_comp.mean().cpu()),
                "hawkes_trigger_neg_comp_mean": float(valid_neg_comp.mean().cpu()),
                "hawkes_trigger_valid_ratio": float(mask_bool.float().mean().cpu()),
            }
        return weight, stats

    def compute_legacy_interaction_loss(
        self,
        s_nodes: torch.Tensor,
        s_node_emb: torch.Tensor,
        t_node_emb: torch.Tensor,
        n_node_emb: torch.Tensor,
        h_node_emb: torch.Tensor,
        t_times: torch.Tensor,
        h_times: torch.Tensor,
        h_time_mask: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        att_score = -((s_node_emb.unsqueeze(1) - h_node_emb) ** 2).sum(dim=2)
        masked_score = torch.where(
            h_time_mask > 0, att_score, torch.full_like(att_score, -1e30)
        )
        att = torch.softmax(masked_score, dim=1)
        att = torch.where(torch.isnan(att), torch.zeros_like(att), att)

        time_decay = self._time_weight(s_nodes, t_times, h_times, h_time_mask)

        p_mu = -((s_node_emb - t_node_emb) ** 2).sum(dim=1)
        p_alpha = -((h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2)

        n_mu = -((s_node_emb.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2)
        n_alpha = -((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1)) ** 2).sum(dim=3)

        hawkes_trigger_weight, hawkes_trigger_stats = (
            self.compute_fuzzy_hawkes_triggering_weights(
                p_alpha=p_alpha,
                n_alpha=n_alpha,
                att=att,
                h_time_mask=h_time_mask,
            )
        )
        p_lambda = p_mu + (att * p_alpha * time_decay * hawkes_trigger_weight).sum(
            dim=1
        )
        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * time_decay.unsqueeze(2)).sum(
            dim=1
        )

        pos_loss = F.softplus(-p_lambda)
        neg_loss_each = F.softplus(n_lambda)

        weighted_pos_loss = (
            pos_weight.view_as(pos_loss) * pos_loss
            if pos_weight is not None
            else pos_loss
        )

        neg_loss = neg_loss_each.sum(dim=1)
        loss_per_sample = weighted_pos_loss + neg_loss

        if self.legacy_loss_reduction == "sum":
            interaction_loss = loss_per_sample.sum()
        else:
            interaction_loss = loss_per_sample.mean()

        stats = {
            "legacy_interaction_loss": float(interaction_loss.detach().cpu()),
            "legacy_interaction_loss_mean": float(
                loss_per_sample.mean().detach().cpu()
            ),
            "legacy_pos_loss": float(pos_loss.mean().detach().cpu()),
            "legacy_neg_loss": float(neg_loss.mean().detach().cpu()),
            "weighted_pos_loss": float(weighted_pos_loss.mean().detach().cpu()),
        }
        stats.update(hawkes_trigger_stats)
        return interaction_loss, stats

    def forward(
        self,
        s_nodes: torch.Tensor,
        t_nodes: torch.Tensor,
        t_times: torch.Tensor,
        n_nodes: torch.Tensor,
        h_nodes: torch.Tensor,
        h_times: torch.Tensor,
        h_time_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = s_nodes.size(0)
        s_node_emb = self.node_emb.index_select(0, s_nodes.view(-1)).view(
            batch_size, -1
        )
        t_node_emb = self.node_emb.index_select(0, t_nodes.view(-1)).view(
            batch_size, -1
        )
        h_node_emb = self.node_emb.index_select(0, h_nodes.view(-1)).view(
            batch_size, self.hist_len, -1
        )
        n_node_emb = self.node_emb.index_select(0, n_nodes.view(-1)).view(
            batch_size, self.neg_size, -1
        )

        current_strength = self.get_fuzzy_weight_strength()

        if self.use_fuzzy_weighting:
            if self.detach_fuzzy_weight:
                with torch.no_grad():
                    r_pos, alpha_pos = self.compute_fuzzy_temporal_relation(
                        s_node_emb.detach(),
                        t_node_emb.detach(),
                        h_node_emb.detach(),
                        t_times,
                        h_times,
                        h_time_mask,
                    )
            else:
                r_pos, alpha_pos = self.compute_fuzzy_temporal_relation(
                    s_node_emb, t_node_emb, h_node_emb, t_times, h_times, h_time_mask
                )
            relation_stats = {
                "r_pos_mean": float(r_pos.detach().mean().cpu()),
                "history_valid_ratio": float(
                    (h_time_mask.sum(dim=1) > 0).float().mean().detach().cpu()
                ),
            }
            reliability, reliability_stats = self.compute_history_reliability(
                alpha_pos, h_time_mask
            )
        else:
            r_pos = None
            reliability = None
            relation_stats = {}
            reliability_stats = {
                "rel_gate_mean": 1.0,
                "rel_gate_std": 0.0,
                "rel_gate_min": 1.0,
                "rel_gate_max": 1.0,
            }

        pos_weight, weight_stats = self.compute_fuzzy_adaptive_weights(
            r_pos, current_strength, reliability
        )

        interaction_loss, legacy_stats = self.compute_legacy_interaction_loss(
            s_nodes=s_nodes,
            s_node_emb=s_node_emb,
            t_node_emb=t_node_emb,
            n_node_emb=n_node_emb,
            h_node_emb=h_node_emb,
            t_times=t_times,
            h_times=h_times,
            h_time_mask=h_time_mask,
            pos_weight=pos_weight,
        )
        original_kl_loss, original_kl_stats = self.compute_original_kl_loss(s_node_emb)
        total_loss = interaction_loss + original_kl_loss

        with torch.no_grad():
            legacy_mean = torch.tensor(
                legacy_stats["legacy_interaction_loss_mean"],
                dtype=torch.float32,
                device=self.device,
            )
            original_kl_ratio = torch.abs(original_kl_loss.detach()) / (
                torch.abs(interaction_loss.detach()) + self.eps
            )
            original_kl_to_mean = torch.abs(original_kl_loss.detach()) / (
                torch.abs(legacy_mean) + self.eps
            )

        stats = {
            "loss_mean": float(total_loss.detach().cpu()),
            "interaction_loss_mean": float(interaction_loss.detach().cpu()),
            "original_kl_loss": float(original_kl_loss.detach().cpu()),
            "original_kl_ratio": float(original_kl_ratio.detach().cpu()),
            "original_kl_to_legacy_mean": float(original_kl_to_mean.detach().cpu()),
            "current_fuzzy_weight_strength": float(current_strength),
        }
        stats.update(legacy_stats)
        stats.update(original_kl_stats)
        stats.update(weight_stats)
        stats.update(relation_stats)
        stats.update(reliability_stats)
        return total_loss, stats

    def update(
        self,
        s_nodes: torch.Tensor,
        t_nodes: torch.Tensor,
        t_times: torch.Tensor,
        n_nodes: torch.Tensor,
        h_nodes: torch.Tensor,
        h_times: torch.Tensor,
        h_time_mask: torch.Tensor,
    ) -> Dict[str, float]:
        self.opt.zero_grad(set_to_none=True)
        loss, stats = self.forward(
            s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected: {loss.item()}")
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            if not torch.isfinite(self.node_emb).all():
                raise RuntimeError("Non-finite parameter detected: node_emb")
            if not torch.isfinite(self.delta).all():
                raise RuntimeError("Non-finite parameter detected: delta")
            if not torch.isfinite(self.cluster_layer).all():
                raise RuntimeError("Non-finite parameter detected: cluster_layer")

        self.running_loss += float(loss.detach().cpu())
        return stats

    def _to_device_batch(self, sample_batched: Dict[str, torch.Tensor]):
        return (
            sample_batched["source_node"].to(
                self.device, dtype=torch.long, non_blocking=True
            ),
            sample_batched["target_node"].to(
                self.device, dtype=torch.long, non_blocking=True
            ),
            sample_batched["target_time"].to(
                self.device, dtype=torch.float32, non_blocking=True
            ),
            sample_batched["neg_nodes"].to(
                self.device, dtype=torch.long, non_blocking=True
            ),
            sample_batched["history_nodes"].to(
                self.device, dtype=torch.long, non_blocking=True
            ),
            sample_batched["history_times"].to(
                self.device, dtype=torch.float32, non_blocking=True
            ),
            sample_batched["history_masks"].to(
                self.device, dtype=torch.float32, non_blocking=True
            ),
        )

    def train(self) -> None:
        num_workers = int(getattr(self.args, "num_workers", 8))
        pin_memory = torch.cuda.is_available()
        persistent_workers = num_workers > 0

        loader = DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,
        )

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.apply_lr_schedule(epoch)
            t0 = time.perf_counter()
            self.running_loss = 0.0
            num_batches = 0

            for sample_batched in loader:
                self.update(*self._to_device_batch(sample_batched))
                num_batches += 1

            if self.the_data in {"arxivLarge", "arxivPhy", "arxivMath"}:
                acc, nmi, ari, f1 = 0.0, 0.0, 0.0, 0.0
            else:
                acc, nmi, ari, f1 = eva(
                    self.clusters, self.labels, self.node_emb, seed=self.seed
                )
                self._update_best_metrics(epoch=epoch, acc=acc, nmi=nmi, ari=ari, f1=f1)

            if self.legacy_loss_reduction == "sum":
                epoch_loss = self.running_loss / max(len(self.data), 1)
            else:
                epoch_loss = self.running_loss / max(num_batches, 1)

            avg = (nmi + ari + acc + f1) / 4.0
            msg = (
                f"epoch {epoch}: loss={epoch_loss:.4f}  "
                f"NMI({nmi:.4f}) ARI({ari:.4f}) ACC({acc:.4f}) F1({f1:.4f}) AVG({avg:.4f})  "
                f"BestAVG({self.best_records['avg']['avg']:.4f}@{int(self.best_records['avg']['epoch'])}) "
                f"BestNMI({self.best_records['nmi']['nmi']:.4f}@{int(self.best_records['nmi']['epoch'])}) "
                f"BestARI({self.best_records['ari']['ari']:.4f}@{int(self.best_records['ari']['epoch'])}) "
                f"BestACC({self.best_records['acc']['acc']:.4f}@{int(self.best_records['acc']['epoch'])}) "
                f"BestF1({self.best_records['f1']['f1']:.4f}@{int(self.best_records['f1']['epoch'])}) "
                f"time={time.perf_counter() - t0:.2f}s\n"
            )
            sys.stdout.write(msg)
            sys.stdout.flush()

        print(self._format_best_record("nmi", self.best_records["nmi"]))
        print(self._format_best_record("ari", self.best_records["ari"]))
        print(self._format_best_record("acc", self.best_records["acc"]))
        print(self._format_best_record("f1", self.best_records["f1"]))
        print(self._format_best_record("avg", self.best_records["avg"]))
        print(
            "Primary Best by Average: NMI(%.4f) ARI(%.4f) ACC(%.4f) F1(%.4f) AVG(%.4f)"
            % (self.best_nmi, self.best_ari, self.best_acc, self.best_f1, self.best_avg)
        )
