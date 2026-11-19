"""Content deduplication, leakage checks, and generator-disjoint splits."""

import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageOps

SPLITS = ("train", "val", "seen_test", "unseen_test")
FIELDS = ("path", "label", "generator", "split", "source_split", "width",
          "height", "format", "sha256", "pixel_sha256")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=FIELDS):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_image(item):
    root, path = item
    relative = path.relative_to(root)
    generator, source_split, folder, _ = relative.parts
    expected = "nature" if generator == "Real" else "ai"
    if folder != expected or source_split not in ("train", "val"):
        raise ValueError(f"Unexpected layout: {relative}")
    content = path.read_bytes()
    with Image.open(io.BytesIO(content)) as original:
        original.load()  # Truncated/undecodable images must stop preparation.
        image_format = original.format
        image = ImageOps.exif_transpose(original).convert("RGB")
        pixel_hash = hashlib.sha256(
            f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
    return dict(path=relative.as_posix(), label=int(generator != "Real"),
                generator=generator, split="", source_split=source_split,
                width=image.width, height=image.height, format=image_format,
                sha256=hashlib.sha256(content).hexdigest(), pixel_sha256=pixel_hash)


def deduplicate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["pixel_sha256"]].append(row)
    kept, duplicates = [], []
    for group in groups.values():
        if len({int(row["label"]) for row in group}) != 1:
            raise ValueError(f"Conflicting real/fake labels: {[r['path'] for r in group]}")
        if len({row["generator"] for row in group}) != 1:
            raise ValueError(f"Identical content has different sources: {[r['path'] for r in group]}")
        # If a duplicate appears in an original holdout, keep it in that holdout.
        group.sort(key=lambda r: (r["source_split"] != "val", r["path"]))
        kept.append(group[0])
        duplicates.extend(dict(kept=group[0]["path"], discarded=r["path"],
                               pixel_sha256=r["pixel_sha256"]) for r in group[1:])
    return sorted(kept, key=lambda r: r["path"]), duplicates


def assert_disjoint(splits):
    for name, rows in splits.items():
        if not rows or {int(r["label"]) for r in rows} != {0, 1}:
            raise ValueError(f"{name} must contain both real and AI images")
        for field in ("path", "sha256", "pixel_sha256"):
            if any(not r.get(field) for r in rows):
                raise ValueError(f"{name}: missing {field}; regenerate manifests with prepare")
            if len({r[field] for r in rows}) != len(rows):
                raise ValueError(f"{name}: duplicate {field}")
        for row in rows:
            if row["split"] != name or (row["generator"] == "Real") != (int(row["label"]) == 0):
                raise ValueError(f"{name}: inconsistent split/label: {row['path']}")
    for a, b in combinations(splits, 2):
        for field in ("path", "sha256", "pixel_sha256"):
            if {r[field] for r in splits[a]} & {r[field] for r in splits[b]}:
                raise ValueError(f"Data leakage: {a} / {b}, shared {field}")
    seen = {r["generator"] for r in splits["train"] if int(r["label"]) == 1}
    for name in ("val", "seen_test"):
        if {r["generator"] for r in splits[name] if int(r["label"]) == 1} != seen:
            raise ValueError(f"{name}: sources differ from training sources")
    unseen = {r["generator"] for r in splits["unseen_test"] if int(r["label"]) == 1}
    if seen & unseen:
        raise ValueError("Unseen test contains a training generator")


def build_splits(rows, seen, unseen, val_fraction=0.2, unseen_per_generator=1000, seed=42):
    if not 0 < val_fraction < 1 or unseen_per_generator < 1:
        raise ValueError("Invalid validation fraction or unseen sample count")
    if not seen or not unseen or len(set(seen + unseen)) != len(seen + unseen) or "Real" in seen + unseen:
        raise ValueError("Seen/unseen generator lists must be nonempty and disjoint")
    rng = random.Random(seed)
    result = {name: [] for name in SPLITS}
    for gen in seen:
        train = [dict(r) for r in rows if r["generator"] == gen and r["source_split"] == "train"]
        test = [dict(r) for r in rows if r["generator"] == gen and r["source_split"] == "val"]
        if len(train) < 2 or not test:
            raise ValueError(f"{gen}: need original train and val images")
        rng.shuffle(train)
        n_val = max(1, min(len(train) - 1, round(len(train) * val_fraction)))
        result["val"].extend(train[:n_val])
        result["train"].extend(train[n_val:])
        result["seen_test"].extend(test)
    for gen in unseen:
        pool = [dict(r) for r in rows if r["generator"] == gen]
        if len(pool) < unseen_per_generator:
            raise ValueError(f"{gen}: requested {unseen_per_generator}, available {len(pool)}")
        rng.shuffle(pool)
        result["unseen_test"].extend(pool[:unseen_per_generator])
    # One global real pool, sampled without replacement across ALL splits.
    real = [dict(r) for r in rows if r["generator"] == "Real"]
    rng.shuffle(real)
    required = sum(len(rs) for rs in result.values())
    if len(real) < required:
        raise ValueError(f"Need {required} unique real images; available {len(real)}")
    offset = 0
    for name, rs in result.items():
        n = len(rs)
        rs.extend(real[offset:offset + n])
        offset += n
        for row in rs:
            row["split"] = name
        rng.shuffle(rs)
    assert_disjoint(result)
    return result


def prepare(args):
    root = Path(args.data_root).resolve()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new output directory")
    generators = ["Real"] + list(args.seen_generators) + list(args.unseen_generators)
    files = []
    for gen in generators:
        directory = root / gen
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files.extend(sorted(p for p in directory.rglob("*") if p.is_file()
                            and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")))
    if not files:
        raise ValueError(f"No images under {root}")
    print(f"Decoding and hashing {len(files)} images...", flush=True)
    with ThreadPoolExecutor(max_workers=args.hash_workers) as pool:
        rows = list(pool.map(inspect_image, ((root, p) for p in files)))
    unique, duplicates = deduplicate(rows)
    splits = build_splits(unique, list(args.seen_generators), list(args.unseen_generators),
                          args.val_fraction, args.unseen_per_generator, args.seed)
    output.mkdir(parents=True)
    for name, rs in splits.items():
        write_csv(output / f"{name}.csv", rs)
    selected = {r["path"] for rs in splits.values() for r in rs}
    write_csv(output / "unused.csv", [r for r in unique if r["path"] not in selected])
    write_csv(output / "duplicates.csv", duplicates, ("kept", "discarded", "pixel_sha256"))
    metadata = dict(schema_version=2, data_root=str(root), seed=args.seed,
                    seen_generators=list(args.seen_generators), unseen_generators=list(args.unseen_generators),
                    val_fraction=args.val_fraction, unseen_per_generator=args.unseen_per_generator,
                    original_files=len(rows), unique_files=len(unique), duplicate_files=len(duplicates),
                    unused_files=len(unique) - len(selected),
                    policy="Seen source train -> train/val; source val -> seen_test. Unseen sources never used for fitting/tuning. Global real pool without replacement. Exact RGB-content deduplication; near-duplicates not detected.",
                    counts={name: dict(Counter(f"{r['generator']}/{r['label']}" for r in rs))
                            for name, rs in splits.items()},
                    manifest_sha256={name: file_hash(output / f"{name}.csv") for name in SPLITS})
    write_json(output / "split_config.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def load_manifests(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "split_config.json").read_text())
    if metadata.get("schema_version") == 3:
        names = tuple(metadata["manifest_sha256"])
        fingerprints = {name: file_hash(directory / f"{name}.csv") for name in names}
        if fingerprints != metadata["manifest_sha256"]:
            raise ValueError("Manifests changed since prepare")
        splits = {name: read_csv(directory / f"{name}.csv") for name in names}
        validate_v3(splits)
        return splits, fingerprints
    if metadata.get("schema_version") != 2:
        raise ValueError("Expected v2 manifests; run prepare first")
    fingerprints = {name: file_hash(directory / f"{name}.csv") for name in SPLITS}
    if fingerprints != metadata["manifest_sha256"]:
        raise ValueError("Manifests changed since prepare; regenerate instead of editing CSVs")
    splits = {name: read_csv(directory / f"{name}.csv") for name in SPLITS}
    assert_disjoint(splits)
    return splits, fingerprints


def validate_v3(splits):
    """Check pure test pools as well as fitting/dev source isolation."""
    expected = {"train", "val", "dev", "seen_test", "unseen_test", "real_test"}
    if set(splits) != expected:
        raise ValueError("Incomplete v3 split set")
    seen = {"BigGAN", "SD15"}
    used = {field: set() for field in ("path", "sha256", "pixel_sha256", "original_id", "group_id")}
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty {name}")
        for row in rows:
            label = int(row["label"])
            if label not in (0, 1) or (row["generator"] == "Real") != (label == 0):
                raise ValueError(f"Invalid v3 label: {row['path']}")
            if row["split"] != name:
                raise ValueError("Incorrect split field")
            if row["source_split"] != ("train" if name in ("train", "val", "dev") else "val"):
                raise ValueError("Official split leakage")
            if name in ("train", "val", "dev", "seen_test") and label and row["generator"] not in seen:
                raise ValueError("Unknown generator leaked into development")
            if name == "unseen_test" and (not label or row["generator"] in seen):
                raise ValueError("Invalid unseen test pool")
            if name == "real_test" and label:
                raise ValueError("Invalid real test pool")
            if name == "seen_test" and not label:
                raise ValueError("Seen test must use shared real_test")
            for field, values in used.items():
                value = row.get(field)
                if not value or value in values:
                    raise ValueError(f"v3 duplicate/missing {field}: {row['path']}")
                values.add(value)
        if name in ("train", "val", "dev") and {int(r["label"]) for r in rows} != {0, 1}:
            raise ValueError(f"{name} requires both classes")


def limit_rows(rows, limit, seed):
    """Optional smoke-test cap, keeping both binary classes balanced."""
    if limit is None or limit >= len(rows):
        return rows
    if limit < 2:
        raise ValueError("Sample cap must be at least 2")
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        pool = [r for r in rows if int(r["label"]) == label]
        rng.shuffle(pool)
        selected.extend(pool[:limit // 2])
    rng.shuffle(selected)
    return selected


def load_supplement(directory, base_splits, base_fingerprints):
    """Verify an independently frozen final-only extension against its base."""
    directory = Path(directory)
    metadata = json.loads((directory / "supplement_config.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("kind") != "frozen_v3_supplement":
        raise ValueError("Unsupported supplemental protocol")
    if metadata["base_manifest_sha256"] != base_fingerprints:
        raise ValueError("Supplement belongs to different base manifests")
    expected = {"midjourney_test", "external_real_test"}
    if set(metadata["manifest_sha256"]) != expected:
        raise ValueError("Incomplete supplemental tests")
    fingerprints = {n: file_hash(directory / f"{n}.csv") for n in expected}
    if fingerprints != metadata["manifest_sha256"]:
        raise ValueError("Supplement manifests changed")
    rows = {n: read_csv(directory / f"{n}.csv") for n in expected}
    used = {k: {r[k] for rs in base_splits.values() for r in rs}
            for k in ("sha256", "pixel_sha256", "original_id", "group_id")}
    root = Path(metadata["data_root"]).resolve()
    paths = set()
    for name, rs in rows.items():
        if not rs or len(rs) != metadata["counts"][name]:
            raise ValueError("Supplement counts differ")
        for row in rs:
            label = 1 if name == "midjourney_test" else 0
            if row["split"] != name or row["source_split"] != "val" or int(row["label"]) != label:
                raise ValueError("Invalid supplemental label/split")
            if row["generator"] != ("Midjourney" if label else "Real"):
                raise ValueError("Invalid supplemental generator")
            path = Path(row["path"])
            if path.is_absolute() or not (root / path).resolve().is_relative_to(root) or row["path"] in paths:
                raise ValueError("Unsafe/duplicate supplemental path")
            paths.add(row["path"])
            for key, values in used.items():
                value = row.get(key)
                if not value or value in values:
                    raise ValueError(f"Supplement leakage: {key}")
                values.add(value)
    return rows, metadata, fingerprints
