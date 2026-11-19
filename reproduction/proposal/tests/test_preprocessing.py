import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from reproduction.proposal.models import Transform, resize_small_crop
from reproduction.p1.models import Transform as HistoricalTransform


class PreprocessingTests(unittest.TestCase):
    def test_no_padding_for_small_or_narrow_images(self):
        for size in [(128, 128), (80, 320), (320, 80), (224, 224), (512, 512)]:
            for train in (False, True):
                out = resize_small_crop(Image.new('RGB', size, (127, 91, 53)),
                                        train, random.Random(42))
                self.assertEqual(out.size, (224, 224))
                self.assertTrue(np.all(np.asarray(out) == [127, 91, 53]))

    def test_large_image_is_cropped_without_resizing(self):
        arr = np.random.default_rng(1).integers(0, 256, (300, 400, 3), dtype=np.uint8)
        out = resize_small_crop(Image.fromarray(arr), False, random.Random(1))
        np.testing.assert_array_equal(np.asarray(out), arr[38:262, 88:312])

    def test_core_methods_share_geometry_and_eval_is_deterministic(self):
        img = Image.fromarray(np.random.default_rng(1).integers(0, 256, (128, 160, 3), dtype=np.uint8))
        for train in (False, True):
            frozen = Transform('frozen_L14_resize', train, {})
            effort = Transform('effort', train, {})
            torch.testing.assert_close(frozen(img, 42), effort(img, 42), rtol=0, atol=0)
        base = Transform('effort', False, {})
        aug = Transform('effort_degradation', False, {})
        torch.testing.assert_close(base(img, 1), aug(img, 99), rtol=0, atol=0)

    def test_effort_matches_old_pipeline_except_padding(self):
        # Replace only historical pad_crop with new geometry; all nine degradation
        # conditions, flip and crop must then match pixel-for-pixel.
        norm = Transform('effort', True, {}).norm
        native = SimpleNamespace(NORM=norm, pad_crop=lambda img, rand: resize_small_crop(img, rand, random))
        img = Image.fromarray(np.random.default_rng(2).integers(0, 256, (128, 160, 3), dtype=np.uint8))
        for method in ('effort', 'effort_degradation'):
            old = HistoricalTransform(method, True, {'kind': 'two_class', 'native': native})
            new = Transform(method, True, {})
            for seed in range(100):
                random.seed(seed)
                torch.testing.assert_close(old(img, seed), new(img, seed), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
