import copy
import unittest
import numpy as np
from genimage_detector.data import validate_v3
from genimage_detector.metrics import choose_fpr_threshold


def fixture():
    result={}
    for name in ('train','val','dev','seen_test','unseen_test','real_test'):
        generators={'seen_test':['BigGAN'],'unseen_test':['VQDM'],'real_test':['Real']}.get(name,['Real','BigGAN'])
        result[name]=[]
        for gen in generators:
            key=name+'/'+gen
            result[name].append(dict(path=key,label=int(gen!='Real'),generator=gen,split=name,
                                    source_split='train' if name in ('train','val','dev') else 'val',
                                    sha256=key,pixel_sha256=key,original_id=key,group_id=key))
    return result


class ProtocolV3Tests(unittest.TestCase):
    def test_shared_real_holdout_and_dev_validate(self):
        validate_v3(fixture())

    def test_official_val_and_derivative_leakage_rejected(self):
        for field,value in [('source_split','val'),('group_id','val/Real'),('pixel_sha256','unseen_test/VQDM')]:
            data=fixture();data['train'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_v3(data)

    def test_unknown_generator_cannot_enter_dev(self):
        data=fixture();data['dev'][1]['generator']='VQDM'
        with self.assertRaises(ValueError):validate_v3(data)

    def test_tied_fpr_boundary_is_conservative(self):
        scores=[0.8]*3+[0.1]*97+[0.9]
        labels=[0]*100+[1]
        t=choose_fpr_threshold(labels,scores,.01)
        self.assertGreater(t,.8)
        self.assertLessEqual(np.mean(np.asarray(scores[:100])>=t),.01)
        self.assertLess(t,.9)

    def test_saturation_and_small_validation_pool(self):
        self.assertGreater(choose_fpr_threshold([0,1],[1.,1.]),1.)
        t=choose_fpr_threshold([0,0,1],[.1,.2,.9])
        self.assertGreater(t,.2)

if __name__=='__main__':unittest.main()
