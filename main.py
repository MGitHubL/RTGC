import argparse
import datetime
import os
import random
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import torch
from model import train

DATASET_CONFIGS = {
    "arxivAI": {"clusters": 5, "membership_kl_beta": 0.50},
    "arxivCS": {"clusters": 40},
    "arxivPhy": {"clusters": 53},
    "arxivMath": {"clusters": 31},
    "arxivLarge": {"clusters": 172},
    "school": {"clusters": 9},
    "dblp": {"clusters": 10, "fuzzy_weight_strength": 0.01, "membership_kl_beta": 0.10},
    "brain": {"clusters": 10},
    "patent": {"clusters": 6},
    "zhihu-answer": {
        "clusters": 50,
        "fuzzy_weight_strength": 0.10,
        "membership_kl_beta": 0.30,
        "fuzzy_hawkes_triggering_strength": 0.00,
    },
    "zhihu-question": {"clusters": 50, "fuzzy_weight_strength": 0.10},
    "tax51": {"clusters": 50},
    "BDSCSocial": {"clusters": 10, "use_lr_decay": 0},
    "Flight-201910": {"clusters": 10, "use_lr_decay": 0},
    "Flight-202004": {
        "clusters": 10,
        "fuzzy_weight_strength": 0.01,
        "fuzzy_hawkes_triggering_strength": 0.01,
    },
    "Flight-202210": {"clusters": 10},
}

MODEL_DEFAULTS = {
    "fuzzy_weight_strength": 0.02,
    "membership_kl_beta": 0.20,
    "fuzzy_hawkes_triggering_strength": 0.02,
    "use_lr_decay": 1,
}


ABLATION_OVERRIDES = {
    "full": {},
    "tgc": {
        "fuzzy_weight_strength": 0.0,
        "membership_kl_beta": 0.0,
        "fuzzy_hawkes_triggering_strength": 0.0,
    },
    "without_interaction_fuzziness": {
        "fuzzy_weight_strength": 0.0,
    },
    "without_membership_fuzziness": {
        "membership_kl_beta": 0.0,
    },
    "without_triggering_fuzziness": {
        "fuzzy_hawkes_triggering_strength": 0.0,
    },
}


def apply_dataset_specific_settings(args: argparse.Namespace) -> None:
    config = DATASET_CONFIGS[args.dataset]
    for name, default in MODEL_DEFAULTS.items():
        setattr(args, name, config.get(name, default))


def apply_ablation_settings(args: argparse.Namespace) -> None:
    for parameter_name, value in ABLATION_OVERRIDES[args.ablation].items():
        setattr(args, parameter_name, value)


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main_train(args: argparse.Namespace) -> None:
    start = datetime.datetime.now()
    trainer = train.TGC(args)
    trainer.train()
    end = datetime.datetime.now()
    print("Training Complete with Time: %s" % str(end - start))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    training = parser.add_argument_group("Data and training")
    training.add_argument(
        "--dataset", type=str, default="patent", choices=list(DATASET_CONFIGS)
    )
    training.add_argument(
        "--ablation", type=str, default="full", choices=list(ABLATION_OVERRIDES.keys())
    )
    training.add_argument("--clusters", type=int, default=None)
    training.add_argument("--epoch", type=int, default=100)
    training.add_argument("--neg_size", type=int, default=10)
    training.add_argument("--hist_len", type=int, default=5)
    training.add_argument("--batch_size", type=int, default=1024)
    training.add_argument("--learning_rate", type=float, default=0.01)
    training.add_argument("--emb_size", type=int, default=128)
    training.add_argument("--directed", type=str2bool, default=False)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    training.add_argument("--num_workers", type=int, default=4)
    training.add_argument("--weight_decay", type=float, default=0.0)

    interaction = parser.add_argument_group("Interaction reliability")
    interaction.add_argument("--beta1", type=float, default=1.0)
    interaction.add_argument("--beta2", type=float, default=1.0)
    interaction.add_argument("--fuzzy_delta", type=float, default=0.1)
    interaction.add_argument("--eta0", type=float, default=1.0)
    interaction.add_argument("--eta1", type=float, default=1.0)
    interaction.add_argument("--normalize_fuzzy_time", type=int, default=1)
    interaction.add_argument("--fuzzy_logit_clamp", type=float, default=60.0)
    interaction.add_argument("--use_fuzzy_weighting", type=int, default=1)
    interaction.add_argument(
        "--fuzzy_weight_strength",
        type=float,
        default=MODEL_DEFAULTS["fuzzy_weight_strength"],
    )
    interaction.add_argument("--warmup_epochs", type=int, default=20)
    interaction.add_argument("--fuzzy_ramp_epochs", type=int, default=30)
    interaction.add_argument("--detach_fuzzy_weight", type=int, default=1)
    interaction.add_argument("--normalize_fuzzy_weight", type=int, default=1)
    interaction.add_argument("--fuzzy_weight_centered", type=int, default=1)
    interaction.add_argument("--fuzzy_weight_clip", type=float, default=2.0)
    interaction.add_argument("--fuzzy_weight_min", type=float, default=0.1)
    interaction.add_argument("--fuzzy_weight_max", type=float, default=10.0)
    interaction.add_argument("--use_reliability_gated_weight", type=int, default=1)
    interaction.add_argument("--reliability_entropy_power", type=float, default=1.0)
    interaction.add_argument("--reliability_min", type=float, default=0.25)
    interaction.add_argument("--reliability_max", type=float, default=1.0)

    history = parser.add_argument_group("Historical reliability")
    history.add_argument("--use_fuzzy_hawkes_triggering", type=int, default=1)
    history.add_argument("--fuzzy_hawkes_triggering_warmup_epoch", type=int, default=20)
    history.add_argument(
        "--fuzzy_hawkes_triggering_strength",
        type=float,
        default=MODEL_DEFAULTS["fuzzy_hawkes_triggering_strength"],
    )
    history.add_argument("--fuzzy_hawkes_triggering_tail_clip", type=float, default=1.0)
    history.add_argument("--fuzzy_hawkes_triggering_min", type=float, default=0.98)
    history.add_argument("--fuzzy_hawkes_triggering_max", type=float, default=1.0)
    history.add_argument("--fuzzy_hawkes_triggering_temp", type=float, default=1.0)
    history.add_argument(
        "--fuzzy_hawkes_triggering_comp_weight", type=float, default=1.0
    )
    history.add_argument(
        "--fuzzy_hawkes_triggering_att_weight", type=float, default=0.5
    )
    history.add_argument(
        "--fuzzy_hawkes_triggering_neg_reduce",
        type=str,
        default="mean",
        choices=["mean", "max", "logsumexp"],
    )
    history.add_argument("--fuzzy_hawkes_triggering_detach", type=int, default=1)

    clustering = parser.add_argument_group("Clustering")
    clustering.add_argument("--use_original_kl_loss", type=int, default=1)
    clustering.add_argument(
        "--legacy_loss_reduction", type=str, default="sum", choices=["sum", "mean"]
    )
    clustering.add_argument("--v", type=float, default=1.0)
    clustering.add_argument("--use_membership_entropy_kl_weight", type=int, default=1)
    clustering.add_argument(
        "--membership_kl_beta", type=float, default=MODEL_DEFAULTS["membership_kl_beta"]
    )
    clustering.add_argument("--membership_kl_weight_min", type=float, default=0.8)
    clustering.add_argument("--membership_kl_weight_max", type=float, default=1.2)
    clustering.add_argument("--membership_kl_detach", type=int, default=1)
    clustering.add_argument("--membership_kl_mean_normalize", type=int, default=1)
    clustering.add_argument("--membership_kl_warmup_epoch", type=int, default=20)

    schedule = parser.add_argument_group("Learning rate schedule")
    schedule.add_argument(
        "--use_lr_decay", type=int, default=MODEL_DEFAULTS["use_lr_decay"]
    )
    schedule.add_argument("--lr_decay_epoch", type=int, default=70)
    schedule.add_argument("--lr_decay_gamma", type=float, default=0.3)

    numerics = parser.add_argument_group("Numerical settings")
    numerics.add_argument("--eps", type=float, default=1e-08)
    numerics.add_argument("--time_eps", type=float, default=1e-06)

    return parser


def print_config(args: argparse.Namespace) -> None:
    print(f"Running dataset: {args.dataset} | ablation: {args.ablation}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.clusters is None:
        args.clusters = DATASET_CONFIGS[args.dataset]["clusters"]

    apply_dataset_specific_settings(args)
    apply_ablation_settings(args)

    print_config(args)
    set_global_seed(args.seed)
    main_train(args)
