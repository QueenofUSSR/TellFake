"""Training, checkpointing, fixed-threshold evaluation, and image inference."""

import json
import os
import platform
import random
import shutil
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from .data import (SPLITS, file_hash, limit_rows, load_manifests, load_supplement, write_csv, write_json)
from .images import ImageTransform, ManifestDataset, PairedImageTransform, parse_condition
from .metrics import binary_metrics, choose_threshold, choose_fpr_threshold, grouped_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_model(arch, pretrained=False):
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def resolve_device(request):
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def loader(root, rows, image_size, batch_size, workers, device, seed,
           training=False, augmentation="none", condition="clean", paired=False):
    transform = PairedImageTransform(image_size) if paired else ImageTransform(image_size, training, augmentation, condition)
    dataset = ManifestDataset(root, rows, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training,
                      num_workers=workers, pin_memory=device.type == "cuda",
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed),
                      persistent_workers=workers > 0)


def snapshot(directory, args):
    write_json(directory / "config.json", vars(args))
    runtime = dict(python=platform.python_version(), torch=str(torch.__version__),
                   torchvision=str(torchvision.__version__), pillow=PIL.__version__,
                   numpy=np.__version__, cuda=torch.version.cuda,
                   scikit_learn=version("scikit-learn"), scipy=version("scipy"),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   docker_image=os.getenv("GENIMAGE_RUNTIME_IMAGE"),
                   docker_image_id=os.getenv("GENIMAGE_RUNTIME_ID"))
    write_json(directory / "runtime.json", runtime)
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "genimage_detector").glob("*.py"))
    paths += sorted((root / "configs").glob("*.json"))
    paths += sorted((root / "scripts").glob("*.py"))
    paths += sorted((root / "scripts").glob("*.sh"))
    paths += [root / "docs" / "DATA_V3.md", root / "requirements-local.lock"]
    paths += [root / "Dockerfile", root / "requirements-runtime.txt"]
    hashes = {}
    for path in paths:
        if path.is_file():
            relative = path.relative_to(root)
            destination = directory / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            hashes[str(relative)] = file_hash(path)
    write_json(directory / "source_sha256.json", hashes)
    if hasattr(args, "manifest_dir"):
        destination = directory / "manifests"
        destination.mkdir()
        metadata = json.loads((Path(args.manifest_dir) / "split_config.json").read_text())
        for filename in [f"{name}.csv" for name in metadata["manifest_sha256"]] + ["split_config.json"]:
            shutil.copy2(Path(args.manifest_dir) / filename, destination / filename)


def new_run(root, name):
    if not name:
        name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Run name must be one directory name")
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


@torch.inference_mode()
def infer(model, batches, device, amp):
    model.eval()
    result = dict(labels=[], probabilities=[], paths=[], generators=[])
    total_loss, count = 0.0, 0
    for images, labels, paths, generators in batches:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(images).flatten()
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise ValueError("Nonfinite inference logits/loss")
        result["labels"].extend(labels.int().cpu().tolist())
        result["probabilities"].extend(logits.float().sigmoid().cpu().tolist())
        result["paths"].extend(paths)
        result["generators"].extend(generators)
        total_loss += float(loss) * len(labels)
        count += len(labels)
    result["loss"] = total_loss / count
    return result


def save_predictions(path, result, threshold):
    rows = [dict(path=p, generator=g, label=y, p_ai=prob,
                 prediction=int(prob >= threshold))
            for p, g, y, prob in zip(result["paths"], result["generators"],
                                    result["labels"], result["probabilities"])]
    write_csv(path, rows, ("path", "generator", "label", "p_ai", "prediction"))


def paired_loss(first, second, labels, weight):
    """Mean two-view BCE + weight * symmetric Bernoulli Jensen-Shannon.

    Log-space mixture and float32 avoid saturation/AMP log(0); both views
    receive gradients. At weight=0 the objective is exactly control B.
    """
    first, second = first.float(), second.float()
    classification = (nn.functional.binary_cross_entropy_with_logits(first, labels) +
                      nn.functional.binary_cross_entropy_with_logits(second, labels)) / 2
    p = torch.stack((nn.functional.logsigmoid(-first), nn.functional.logsigmoid(first)), dim=1)
    q = torch.stack((nn.functional.logsigmoid(-second), nn.functional.logsigmoid(second)), dim=1)
    mixture = torch.logaddexp(p, q) - np.log(2.)
    consistency = ((p.exp() * (p-mixture)).sum(1) + (q.exp() * (q-mixture)).sum(1)).mean() / 2
    consistency = consistency.clamp_min(0)
    return classification + weight * consistency, classification, consistency


def finite_amp_step(model, optimizer, scaler, closure, max_retries=8):
    """Replay the same batch after AMP overflow; never drop an optimizer update."""
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None
    for attempt in range(max_retries + 1):
        if attempt:
            with torch.no_grad():
                for n, b in model.named_buffers(): b.copy_(buffers[n])
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng)
        optimizer.zero_grad(set_to_none=True)
        values = closure()
        if not torch.isfinite(values[0]): raise ValueError("Nonfinite forward loss")
        scaler.scale(values[0]).backward()
        scaler.unscale_(optimizer)
        norm = torch.stack([p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]).norm()
        if torch.isfinite(norm):
            scaler.step(optimizer); scaler.update()
            return values, float(norm), attempt
        if not scaler.is_enabled() or attempt == max_retries:
            raise ValueError("Nonfinite gradients persisted after AMP replay")
        scaler.update(new_scale=scaler.get_scale() / 2)
    raise AssertionError("unreachable")


def train(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    splits, fingerprints = load_manifests(args.manifest_dir)
    train_rows = limit_rows(splits["train"], args.max_train_samples, args.seed)
    val_rows = limit_rows(splits["val"], args.max_val_samples, args.seed)
    model = build_model(args.arch, args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    train_loader = loader(args.data_root, train_rows, args.image_size, args.batch_size,
                          args.workers, device, args.seed, True, args.augmentation, paired=args.method != "A")
    val_loader = loader(args.data_root, val_rows, args.image_size, args.batch_size,
                        args.workers, device, args.seed)
    if args.name and (Path(args.checkpoint_root) / args.name).exists():
        raise FileExistsError(Path(args.checkpoint_root) / args.name)
    logs = new_run(args.log_root, args.name)
    checkpoints = new_run(args.checkpoint_root, logs.name)
    snapshot(logs, args)
    write_csv(logs / "train_used.csv", train_rows, tuple(train_rows[0]))
    write_csv(logs / "val_used.csv", val_rows, tuple(val_rows[0]))
    write_json(logs / "manifest_sha256.json", fingerprints)
    smoke = bool(args.max_train_samples or args.max_val_samples)
    print(f"Run={logs.name} device={device} train={len(train_rows)} val={len(val_rows)} smoke={smoke}", flush=True)
    best = (-float("inf"), -float("inf"))
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        total_loss, total_classification, total_consistency, count = 0.0, 0.0, 0.0, 0
        first_gradient_norm = None
        amp_retries = 0
        for images, labels, _, _ in train_loader:
            labels = labels.to(device, non_blocking=True)
            images = (torch.cat(images, dim=0) if args.method != "A" else images).to(device, non_blocking=True)
            def closure():
                with torch.amp.autocast(device_type=device.type, enabled=amp):
                    logits = model(images).flatten()
                    if args.method == "A":
                        classification = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                        return classification, classification, classification.new_zeros(())
                    first, second = logits.split(len(labels))
                    return paired_loss(first, second, labels, args.consistency_weight)
            (loss, classification, consistency), gradient_norm, retries = finite_amp_step(model, optimizer, scaler, closure)
            amp_retries += retries
            if retries:
                print(json.dumps(dict(event="amp_batch_replay", epoch=epoch, batch=count // args.batch_size, retries=retries, scale=scaler.get_scale())), flush=True)
            if first_gradient_norm is None: first_gradient_norm = gradient_norm
            total_loss += float(loss.detach()) * len(labels)
            total_classification += float(classification.detach()) * len(labels)
            total_consistency += float(consistency.detach()) * len(labels)
            count += len(labels)
        train_seconds = time.perf_counter() - start
        validation = infer(model, val_loader, device, amp)
        validation_seconds = time.perf_counter() - start - train_seconds
        threshold = (choose_fpr_threshold(validation["labels"], validation["probabilities"], args.target_fpr)
                     if args.target_fpr is not None else choose_threshold(validation["labels"], validation["probabilities"]))
        metrics = binary_metrics(validation["labels"], validation["probabilities"], threshold)
        record = dict(epoch=epoch, train_loss=total_loss / count, val_loss=validation["loss"],
                      lr=optimizer.param_groups[0]["lr"], seconds=time.perf_counter() - start,
                      validation=metrics)
        record.update(updates=len(train_loader), peak_cuda_allocated_bytes=(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0))
        record.update(train_seconds=train_seconds, validation_seconds=validation_seconds,
                      train_images_per_second=count / train_seconds)
        record.update(classification_loss=total_classification / count,
                      consistency_loss=total_consistency / count,
                      method=args.method, consistency_weight=args.consistency_weight,
                      first_gradient_norm=first_gradient_norm, amp_batch_retries=amp_retries)
        with (logs / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        checkpoint = dict(schema_version=1, arch=args.arch, image_size=args.image_size,
                          state_dict=model.state_dict(), threshold=threshold, epoch=epoch,
                          val_metrics=metrics, manifest_sha256=fingerprints, config=vars(args),
                          smoke_test=smoke,
                          source_sha256=json.loads((logs / "source_sha256.json").read_text()))
        save_checkpoint(checkpoints / "last.pt", checkpoint)
        score = (metrics["auroc"], -validation["loss"])
        if score > best:
            best = score
            save_checkpoint(checkpoints / "best.pt", checkpoint)
            save_predictions(logs / "best_val_predictions.csv", validation, threshold)
            write_json(logs / "best_validation.json", record)
        scheduler.step()
        print(json.dumps(record, allow_nan=False), flush=True)
    print(f"Best checkpoint: {checkpoints / 'best.pt'}", flush=True)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    model = build_model(checkpoint["arch"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def evaluate(args):
    for condition in args.conditions:
        parse_condition(condition)
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Duplicate evaluation conditions")
    set_seed(args.seed)
    device = resolve_device(args.device)
    splits, fingerprints = load_manifests(args.manifest_dir)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if fingerprints != checkpoint["manifest_sha256"]:
        raise ValueError("Checkpoint and evaluation manifest fingerprints differ")
    supplemental_names = {"midjourney_test", "external_real_test"}
    supplemental = None
    if supplemental_names.intersection(args.splits):
        if not args.supplement_dir:
            raise ValueError("Supplemental splits require --supplement-dir")
        supplemental = load_supplement(args.supplement_dir, splits, fingerprints)
        if args.max_samples:
            raise ValueError("Use synthetic tests to check final-only extension; do not sample final tests for development")
    output = new_run(args.log_root, args.name)
    snapshot(output, args)
    if supplemental:
        destination = output / "supplement_manifests"
        shutil.copytree(args.supplement_dir, destination)
    threshold = checkpoint["threshold"]  # Never refit on test images.
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()),
                  checkpoint_sha256=file_hash(args.checkpoint), threshold=threshold,
                  manifest_sha256=fingerprints,
                  smoke_test=bool(checkpoint.get("smoke_test") or args.max_samples),
                  protocol="Frozen validation threshold. Perturb before fixed input resize; both classes receive identical conditions. Each generator is paired with the same real pool.",
                  results={})
    summary = []
    if supplemental:
        report["supplement_manifest_sha256"] = supplemental[2]
        report["supplement_data_root"] = supplemental[1]["data_root"]
    for split in args.splits:
        evaluation_root = args.data_root
        if split in supplemental_names:
            rows = list(supplemental[0][split])
            evaluation_root = supplemental[1]["data_root"]
            if split == "midjourney_test":
                prefix = Path(args.data_root).resolve().relative_to(Path(evaluation_root).resolve())
                rows += [dict(r, path=str(prefix / r["path"])) for r in splits["real_test"]]
        else:
            rows = limit_rows(splits[split], args.max_samples, args.seed)
        if "real_test" in splits and split in ("seen_test", "unseen_test"):
            # Pure AI source manifests share the same distinct real holdout.
            rows = limit_rows(splits[split] + splits["real_test"], args.max_samples, args.seed)
        report["results"][split] = {}
        for condition in args.conditions:
            batches = loader(evaluation_root, rows, checkpoint["image_size"], args.batch_size,
                             args.workers, device, args.seed, condition=condition)
            predictions = infer(model, batches, device, args.amp)
            metrics = (dict(overall=binary_metrics(predictions["labels"], predictions["probabilities"], threshold),
                            by_generator={}) if split == "external_real_test" else
                       grouped_metrics(predictions["labels"], predictions["probabilities"],
                                       predictions["generators"], threshold))
            if split == "external_real_test":
                # A real-only pool measures FPR, not binary detection recall.
                metrics["overall"].update(balanced_accuracy=None, precision=None, recall=None, f1=None)
            report["results"][split][condition] = metrics
            save_predictions(output / f"{split}_{condition.replace(':', '_')}.csv", predictions, threshold)
            for group, values in [("overall", metrics["overall"])] + list(metrics["by_generator"].items()):
                summary.append(dict(split=split, condition=condition, group=group, **values))
            print(json.dumps(dict(split=split, condition=condition, **metrics["overall"])), flush=True)
            del batches
    write_json(output / "metrics.json", report)
    write_csv(output / "summary.csv", summary, tuple(summary[0]))
    print(f"Evaluation: {output}", flush=True)


@torch.inference_mode()
def predict(args):
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    tensor = ImageTransform(checkpoint["image_size"])(image).unsqueeze(0).to(device)
    with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
        probability = float(model(tensor).flatten().float().sigmoid().item())
    ai = probability >= checkpoint["threshold"]
    result = dict(image=str(args.image), label="AI-Generated" if ai else "Real", p_ai=probability,
                  confidence=probability if ai else 1 - probability,
                  threshold=checkpoint["threshold"], smoke_test=checkpoint.get("smoke_test", False),
                  confidence_note="Uncalibrated model probability, not a guarantee of authenticity")
    if args.output:
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
