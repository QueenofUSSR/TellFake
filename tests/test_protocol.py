import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, PngImagePlugin

from genimage_detector.data import (assert_disjoint, build_splits, deduplicate,
                                   inspect_image, load_manifests, prepare)
from genimage_detector.images import DEFAULT_CONDITIONS, ImageTransform, parse_condition, perturb
from genimage_detector.metrics import binary_metrics, choose_threshold, grouped_metrics


def synthetic_rows():
    rows = []
    for gen in ("Real", "BigGAN", "SD15", "ADM", "GLIDE"):
        for split, count in (("train", 20 if gen == "Real" else 4), ("val", 0 if gen == "Real" else 2)):
            for i in range(count):
                key = f"{gen}/{split}/{i}"
                rows.append(dict(path=key, generator=gen, label=int(gen != "Real"),
                                 source_split=split, split="", sha256="file:" + key,
                                 pixel_sha256="pixel:" + key))
    return rows


class SplitProtocolTests(unittest.TestCase):
    def test_global_real_pool_and_original_holdout(self):
        rows = synthetic_rows()
        result = build_splits(rows, ["BigGAN", "SD15"], ["ADM", "GLIDE"],
                              val_fraction=0.25, unseen_per_generator=2)
        assert_disjoint(result)
        self.assertEqual([len(result[n]) for n in ("train", "val", "seen_test", "unseen_test")], [12, 4, 8, 8])
        for name, selected in result.items():
            self.assertEqual(sum(int(r["label"]) for r in selected), len(selected) // 2)
            for row in selected:
                if row["generator"] in ("BigGAN", "SD15"):
                    self.assertEqual(row["source_split"], "val" if name == "seen_test" else "train")
        self.assertTrue(all(r["split"] == "" for r in rows))
        self.assertEqual(result, build_splits(rows, ["BigGAN", "SD15"], ["ADM", "GLIDE"], 0.25, 2))

    def test_reencoded_identical_pixels_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "Real" / "train" / "nature"
            folder.mkdir(parents=True)
            image = Image.new("RGB", (16, 20), (12, 34, 56))
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("description", "Different bytes, identical pixels")
            image.save(folder / "a.png")
            image.save(folder / "b.png", pnginfo=metadata)
            rows = [inspect_image((root, p)) for p in sorted(folder.glob("*.png"))]
            self.assertNotEqual(rows[0]["sha256"], rows[1]["sha256"])
            kept, duplicates = deduplicate(rows)
            self.assertEqual((len(kept), len(duplicates)), (1, 1))

    def test_conflicting_labels_fail(self):
        rows = synthetic_rows()[:2]
        rows[1].update(generator="BigGAN", label=1, pixel_sha256=rows[0]["pixel_sha256"])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            deduplicate(rows)

    def test_duplicate_with_original_val_stays_in_holdout(self):
        rows = synthetic_rows()[20:22]
        rows[1].update(source_split="val", pixel_sha256=rows[0]["pixel_sha256"])
        kept, _ = deduplicate(rows)
        self.assertEqual(kept[0]["source_split"], "val")

    def test_shared_content_under_different_paths_fails(self):
        result = build_splits(synthetic_rows(), ["BigGAN", "SD15"], ["ADM", "GLIDE"], 0.25, 2)
        result["seen_test"][0]["pixel_sha256"] = result["train"][0]["pixel_sha256"]
        with self.assertRaisesRegex(ValueError, "Data leakage"):
            assert_disjoint(result)

    def test_insufficient_real_images_fail_instead_of_resampling(self):
        rows = [r for r in synthetic_rows() if r["generator"] != "Real"]
        with self.assertRaisesRegex(ValueError, "unique real"):
            build_splits(rows, ["BigGAN", "SD15"], ["ADM", "GLIDE"], 0.25, 2)

    def test_manifest_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "images"
            for i, row in enumerate(synthetic_rows()):
                folder = root / row["generator"] / row["source_split"] / ("nature" if row["label"] == 0 else "ai")
                folder.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (16, 16), (i, i, i)).save(folder / f"{i}.png")
            output = Path(directory) / "manifests"
            args = SimpleNamespace(data_root=str(root), output_dir=str(output),
                                   seen_generators=["BigGAN", "SD15"], unseen_generators=["ADM", "GLIDE"],
                                   hash_workers=2, val_fraction=0.25, unseen_per_generator=2, seed=42)
            prepare(args)
            load_manifests(output)
            with (output / "train.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Manifests changed"):
                load_manifests(output)


class EvaluationProtocolTests(unittest.TestCase):
    def test_threshold_is_selected_using_validation_balanced_accuracy(self):
        labels, probabilities = [0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]
        threshold = choose_threshold(labels, probabilities)
        self.assertAlmostEqual(threshold, 0.3)
        self.assertEqual(binary_metrics(labels, probabilities, threshold)["balanced_accuracy"], 1)

    def test_tied_predictions_are_handled(self):
        threshold = choose_threshold([0, 1], [0.5, 0.5])
        metrics = binary_metrics([0, 1], [0.5, 0.5], threshold)
        self.assertEqual(metrics["auroc"], 0.5)
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (0, 1, 0, 1))

    def test_per_generator_metrics_include_real_images(self):
        result = grouped_metrics([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8],
                                 ["Real", "Real", "ADM", "GLIDE"], 0.5)
        self.assertEqual(set(result["by_generator"]), {"ADM", "GLIDE"})
        self.assertEqual(result["by_generator"]["ADM"]["n_real"], 2)
        self.assertEqual(result["by_generator"]["GLIDE"]["auroc"], 1)

    def test_perturbations_are_deterministic_and_preserve_dimensions(self):
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (80, 120, 3), dtype=np.uint8))
        original = image.tobytes()
        for condition in DEFAULT_CONDITIONS:
            with self.subTest(condition=condition):
                transformed = perturb(image, condition)
                self.assertEqual(transformed.size, image.size)
                self.assertEqual(transformed.tobytes(), perturb(image, condition).tobytes())
                self.assertEqual(tuple(ImageTransform(64, condition=condition)(image).shape), (3, 64, 64))
                if condition != "clean":
                    self.assertNotEqual(transformed.tobytes(), original)
        self.assertEqual(image.tobytes(), original)

    def test_invalid_perturbations_fail(self):
        for condition in ("jpeg:0", "jpeg:99.5", "resize:1", "resize:0", "blur:-1", "blur:nan", "noise:1"):
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                parse_condition(condition)


if __name__ == "__main__":
    unittest.main()
