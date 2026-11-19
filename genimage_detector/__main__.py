import argparse
import json
import os
from pathlib import Path

from .data import prepare


def parser():
    result = argparse.ArgumentParser(description="GenImage subset: prepare, train, evaluate, predict")
    commands = result.add_subparsers(dest="command", required=True)
    dataset_root = os.getenv("GENIMAGE_ROOT", "/mnt/ssd/data/datasets/GenImage/standard_v1")
    log_root = os.getenv("GENIMAGE_LOG_ROOT", "/mnt/ssd/data/logs/arin5201")
    checkpoint_root = os.getenv("GENIMAGE_CHECKPOINT_ROOT", "/mnt/ssd/data/ckpt/arin5201")

    def data_options(p):
        p.add_argument("--data-root", default=dataset_root)
        p.add_argument("--manifest-dir", default="data/manifests/v2")

    def runtime_options(p):
        p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    def batch_options(p):
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--seed", type=int, default=42)

    def log_options(p):
        p.add_argument("--log-root", default=log_root)
        p.add_argument("--name", help="Unique run name; existing directories are never overwritten")

    p = commands.add_parser("prepare", help="Decode, deduplicate and create leakage-free manifests")
    p.add_argument("--data-root", default=dataset_root)
    p.add_argument("--output-dir", default="data/manifests/v2")
    p.add_argument("--seen-generators", nargs="+", default=["BigGAN", "SD15"])
    p.add_argument("--unseen-generators", nargs="+", default=["ADM", "GLIDE"])
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--unseen-per-generator", type=int, default=1000)
    p.add_argument("--hash-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    training = commands.add_parser("train", help="Fine-tune a pretrained ResNet binary classifier")
    data_options(training)
    runtime_options(training)
    batch_options(training)
    log_options(training)
    training.add_argument("--config", help="JSON hyperparameters; command-line options take precedence")
    training.add_argument("--checkpoint-root", default=checkpoint_root)
    training.add_argument("--arch", choices=("resnet50", "resnet18"), default="resnet50")
    training.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--image-size", type=int, default=224)
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--lr", type=float, default=0.0001)
    training.add_argument("--weight-decay", type=float, default=0.0001)
    training.add_argument("--augmentation", choices=("none", "robust", "jpeg", "resize", "blur"), default="none")
    training.add_argument("--target-fpr", type=float, default=None)
    training.add_argument("--method", choices=("A", "B", "C"), default="A")
    training.add_argument("--consistency-weight", type=float, default=0.0)
    training.add_argument("--max-train-samples", type=int, help="Smoke test only; saved in checkpoint")
    training.add_argument("--max-val-samples", type=int, help="Smoke test only; saved in checkpoint")

    p = commands.add_parser("evaluate", help="Seen/unseen tests under fixed perturbation conditions")
    data_options(p)
    runtime_options(p)
    batch_options(p)
    log_options(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--supplement-dir", help="Frozen final-only extension; unused for dev")
    p.add_argument("--splits", nargs="+", choices=("seen_test", "unseen_test", "dev", "midjourney_test", "external_real_test"),
                   default=["seen_test", "unseen_test"])
    p.add_argument("--conditions", nargs="+", default=["clean", "jpeg:90", "jpeg:70", "jpeg:50",
                                                     "resize:0.5", "resize:0.25", "blur:1", "blur:2"])
    p.add_argument("--max-samples", type=int, help="Smoke test cap PER split")

    p = commands.add_parser("predict", help="Classify one JPEG/PNG image")
    runtime_options(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output", help="Optional new JSON output file")
    return result, training


def main():
    main_parser, training_parser = parser()
    args = main_parser.parse_args()
    try:
        if args.command == "train" and args.config:
            config = json.loads(Path(args.config).read_text())
            allowed = {"arch", "pretrained", "image_size", "batch_size", "workers", "epochs",
                       "lr", "weight_decay", "augmentation", "amp", "seed", "target_fpr",
                       "method", "consistency_weight"}
            if not isinstance(config, dict) or set(config) - allowed:
                raise ValueError(f"Config must contain only hyperparameters: {sorted(allowed)}")
            training_parser.set_defaults(**config)
            args = main_parser.parse_args()
        if args.command == "prepare":
            if args.hash_workers < 1:
                raise ValueError("hash-workers must be positive")
            prepare(args)
            return
        for field in ("batch_size", "epochs"):
            if hasattr(args, field) and getattr(args, field) < 1:
                raise ValueError(f"{field} must be positive")
        if hasattr(args, "workers") and args.workers < 0:
            raise ValueError("workers cannot be negative")
        if args.command == "train":
            import math
            if args.method not in ("A", "B", "C") or not math.isfinite(args.consistency_weight) or args.consistency_weight < 0:
                raise ValueError("Invalid method/consistency weight")
            if args.method != "C" and args.consistency_weight != 0:
                raise ValueError("Only method C may have nonzero consistency weight")
            if args.method != "A" and args.augmentation != "none":
                raise ValueError("B/C use the fixed paired view law; augmentation must be none")
            if args.target_fpr is not None and not 0 < args.target_fpr < 1:
                raise ValueError("target-fpr must be between 0 and 1")
            if args.image_size < 64 or args.lr <= 0 or args.weight_decay < 0:
                raise ValueError("Need image-size >= 64, lr > 0, weight-decay >= 0")
            if args.arch not in ("resnet50", "resnet18"):
                raise ValueError("Invalid architecture in config")
            if args.augmentation not in ("none", "robust", "jpeg", "resize", "blur"):
                raise ValueError("Invalid augmentation in config")
        from . import engine
        {"train": engine.train, "evaluate": engine.evaluate, "predict": engine.predict}[args.command](args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        main_parser.error(str(error))


if __name__ == "__main__":
    main()
