import math
import random
import unittest
import numpy as np
import torch
from PIL import Image
from genimage_detector.engine import paired_loss
from genimage_detector.images import PairedImageTransform, perturb, parse_condition


class PairedProtocolTests(unittest.TestCase):
    def test_zero_weight_matches_control_b_value_and_gradients(self):
        a=torch.tensor([-2.,.3,1.2],requires_grad=True)
        b=torch.tensor([.7,-1.,.6],requires_grad=True)
        labels=torch.tensor([0.,1.,1.])
        expected=(torch.nn.functional.binary_cross_entropy_with_logits(a,labels)+torch.nn.functional.binary_cross_entropy_with_logits(b,labels))/2
        reference=torch.autograd.grad(expected,(a,b),retain_graph=True)
        actual,_,_=paired_loss(a,b,labels,0.)
        gradients=torch.autograd.grad(actual,(a,b))
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        for want,got in zip(reference,gradients):torch.testing.assert_close(want,got,rtol=0,atol=0)

    def test_js_is_symmetric_bounded_and_both_views_receive_gradients(self):
        a=torch.tensor([-5.,.3,2.],requires_grad=True);b=torch.tensor([3.,-1.,.6],requires_grad=True)
        y=torch.tensor([0.,1.,1.])
        loss,_,js=paired_loss(a,b,y,.1)
        _,_,reverse=paired_loss(b,a,y,.1)
        torch.testing.assert_close(js,reverse)
        self.assertGreaterEqual(float(js),0);self.assertLessEqual(float(js),math.log(2))
        loss.backward()
        for gradient in (a.grad,b.grad):
            self.assertTrue(torch.isfinite(gradient).all());self.assertTrue(torch.any(gradient!=0))
        _,_,same=paired_loss(a,a,y,1.)
        self.assertAlmostEqual(float(same),0,places=7)

    def test_extreme_logits_are_finite(self):
        a=torch.tensor([-1000.,1000.],requires_grad=True);b=-a.detach().clone().requires_grad_()
        loss,_,js=paired_loss(a,b,torch.tensor([0.,1.]),1.)
        self.assertTrue(torch.isfinite(loss));self.assertAlmostEqual(float(js),math.log(2),places=6)
        loss.backward();self.assertTrue(torch.isfinite(a.grad).all())

    def test_processing_chain_order_and_repeatability(self):
        image=Image.fromarray(np.random.default_rng(42).integers(0,256,(80,120,3),dtype=np.uint8))
        first=perturb(image,'resize:0.5>jpeg:70')
        manual=perturb(perturb(image,'resize:0.5'),'jpeg:70')
        self.assertEqual(first.tobytes(),manual.tobytes())
        self.assertNotEqual(first.tobytes(),perturb(image,'jpeg:70>resize:0.5').tobytes())
        for text in ('jpeg:70>','>clean','resize:1>jpeg:70'):
            with self.assertRaises(ValueError):parse_condition(text)
        transform=PairedImageTransform(64)
        random.seed(42);b=transform(image)
        random.seed(42);c=transform(image)
        for x,y in zip(b,c):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual(tuple(b[0].shape),(3,64,64))

if __name__=='__main__':unittest.main()
