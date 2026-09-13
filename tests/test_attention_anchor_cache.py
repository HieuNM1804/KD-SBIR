"""Cache lifecycle and seen-only fitting, without downloading pretrained weights."""
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.attention_anchor_cache import prepare_anchor_cache, evaluate_teacher, save_atomic
from src.dataset import TrainDataset


class AnchorCacheTests(unittest.TestCase):
    def test_cache_paths_provenance_and_full_teacher_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for mod in ('photo', 'sketch'):
                for label, category in enumerate(('seen_a', 'seen_b', 'unseen')):
                    for index in range(2):
                        path = root/mod/category/f'{index}.png'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.new('RGB', (8,8), (40+label*50, 30+index*40, 80)).save(path)
            args = Namespace(root=str(root),dataset='fixture',seed=42,max_size=8,workers=0,
                             teacher_cache_path=str(root/'teacher.pt'),anchor_cache_path=str(root/'anchors.pt'),
                             anchor_count=4,anchor_temperature=.1,anchor_pooling='attention',
                             anchor_fit_images_per_class=1,anchor_fit_tokens_per_image=2,
                             anchor_fit_iterations=3,anchor_teacher_batch_size=2)
            with patch('src.dataset.UNSEEN_CLASSES', {'fixture':['unseen']}):
                dataset = TrainDataset(args)
            metadata = {'teacher':'fixture'}
            torch.save({'metadata':metadata,'teacher_prompt_state_dict':{'p':torch.ones(1)}},args.teacher_cache_path)
            teacher = Namespace(visual=Namespace(conv1=torch.nn.Conv2d(3,1280,1)))
            generator = torch.Generator().manual_seed(7)
            projection = torch.randn(3,1280,generator=generator)
            def evidence(_teacher, _controller, images, modality):
                tokens = images.flatten(2)[:,:,:4].transpose(1,2) @ projection
                tokens = tokens + torch.arange(4)[None,:,None]*.03
                attention = torch.tensor([.1,.2,.3,.4]).expand(len(images),-1)
                return tokens, attention, tokens.mean(1)
            with patch('src.attention_anchor_cache.load_teacher',return_value=(teacher,None)), \
                 patch('src.attention_anchor_cache.evidence',side_effect=evidence):
                cache = prepare_anchor_cache(args,dataset,metadata)
                self.assertEqual(cache['sketch'].shape,(4,4))
                self.assertFalse(any('unseen' in p for _,p in cache['fit_paths']))
                self.assertLess(Path(args.anchor_cache_path).stat().st_size,100000)
                torch.testing.assert_close(dataset[0][6],cache['sketch'][0])
                self.assertTrue(any(torch.equal(dataset[0][5],p) for p in cache['photo'][:2]))
                anchors = cache['anchors'].clone()
                loader = DataLoader(TensorDataset(torch.randn(4,3,8,8),torch.tensor([0,1,0,1])))
                def metrics(sk,ph,slabels,plabels,_dataset):
                    self.assertEqual(len(sk),4);self.assertEqual(len(ph),4)
                    torch.testing.assert_close(sk.norm(dim=-1),torch.ones(4))
                    return torch.tensor(.5),torch.tensor(.5),None,100
                with patch('src.model._retrieval_metrics',side_effect=metrics) as metric:
                    report = evaluate_teacher(args,cache,loader,loader,root/'report.json')
                self.assertEqual(metric.call_count,3)
                self.assertEqual([r['descriptor'] for r in report['results']],['global','attention','uniform'])
                torch.testing.assert_close(cache['anchors'],anchors,atol=0,rtol=0)
            with patch('src.attention_anchor_cache.load_teacher',side_effect=AssertionError('cache miss')):
                prepare_anchor_cache(args,dataset,metadata)
                args.anchor_temperature=.2
                with self.assertRaisesRegex(ValueError,'provenance'):prepare_anchor_cache(args,dataset,metadata)
                args.anchor_temperature=.1
                Image.new('RGB',(8,8),'red').save(dataset.all_sketches_path[0])
                with self.assertRaisesRegex(ValueError,'provenance'):prepare_anchor_cache(args,dataset,metadata)
            original = Path(args.anchor_cache_path).read_bytes()
            with self.assertRaises(FileExistsError):save_atomic(args.anchor_cache_path,{'x':torch.ones(1)})
            self.assertEqual(Path(args.anchor_cache_path).read_bytes(),original)
            self.assertFalse(list(root.glob('*.tmp')))

    def test_real_openclip_prompt_capture(self):
        from open_clip.transformer import VisionTransformer
        from src.teacher_prompts import TeacherPromptController
        from src.attention_anchor_cache import evidence
        visual = VisionTransformer(image_size=8,patch_size=4,width=64,layers=2,heads=4,
                                   mlp_ratio=2,output_dim=32).eval().requires_grad_(False)
        controller = TeacherPromptController(visual,2,2,.02,42).eval()
        images=torch.randn(2,3,8,8)
        tokens,a,features=evidence(Namespace(visual=visual),controller,images,'sketch')
        self.assertEqual(tokens.shape,(2,4,64))
        torch.testing.assert_close(a.sum(-1),torch.ones(2))
        torch.testing.assert_close(features,controller(images,'sketch'))
        self.assertFalse(tokens.requires_grad)
        self.assertFalse(visual.transformer.resblocks[-1].attn._forward_pre_hooks)


if __name__ == '__main__':unittest.main()
