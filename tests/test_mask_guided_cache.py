from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from src.dataset import TrainDataset,TeacherFeatureDataset
from src.mask_guided import apply_mask
from src.mask_guided_cache import prepare_mask_cache,save_cache


class MaskCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_real_teacher_targets_dataset_mapping_cache_and_metadata(self):
        from open_clip.transformer import VisionTransformer
        from src.teacher_prompts import TeacherPromptController
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for mod in ('sketch','photo'):
                for label,category in enumerate(('seen_a','seen_b','unseen')):
                    for index in range(2):
                        path=root/mod/category/f'{index}.png';path.parent.mkdir(parents=True,exist_ok=True)
                        image=Image.new('RGB',(8,8),(30+label*50,80,40+index*60))
                        image.putpixel((index,label),(255,255,255));image.save(path)
            args=Namespace(root=str(root),dataset='fixture',seed=42,max_size=8,workers=0,
                           teacher_cache_path=str(root/'teacher.pt'),mask_cache_path=str(root/'mask.pt'),
                           mask_ratio=.25,mask_grid=2,mask_teacher_batch_size=2,mask_strategy='attention',lambda_response=1)
            with patch('src.dataset.UNSEEN_CLASSES',{'fixture':['unseen']}):dataset=TrainDataset(args)
            visual=VisionTransformer(image_size=8,patch_size=4,width=64,layers=2,heads=4,mlp_ratio=2,output_dim=1024).eval().requires_grad_(False)
            controller=TeacherPromptController(visual,2,2,.02,42).eval().requires_grad_(False)
            teacher=Namespace(visual=visual);metadata={'fixture':True}
            torch.save({'metadata':metadata,'teacher_prompt_state_dict':controller.state_dict()},args.teacher_cache_path)
            with patch('src.mask_guided_cache.load_teacher',return_value=(teacher,controller)) as build:
                cache=prepare_mask_cache(args,dataset,metadata,root/'report')
            self.assertEqual(build.call_count,1)
            self.assertEqual(len(cache['photo']['full']),4)
            self.assertTrue((root/'report/sketch_mask_examples.png').exists())
            self.assertTrue((root/'report/teacher_mask_audit.json').exists())
            self.assertTrue((root/'report/photo_teacher_mask_samples.csv').exists())
            self.assertFalse(any('unseen' in p for mod in cache['examples'].values() for p in mod['paths']))
            # Targets correspond exactly to the same masked pixels the student receives.
            images=torch.stack([TeacherFeatureDataset(dataset.all_photo_paths,8)[i] for i in range(4)])
            for strategy in ('attention','random'):
                with torch.no_grad():expected=torch.nn.functional.normalize(controller(apply_mask(images,cache['photo'][strategy+'_mask']),'photo').float(),dim=-1)
                torch.testing.assert_close(cache['photo'][strategy+'_target'].float(),expected,atol=.0001,rtol=.001)
            # Verify sampled photo and mask-target pairing for every sample and epoch.
            from src.dataset import sample_seed
            import numpy as np
            for epoch in (0,2):
                for index in range(len(dataset)):
                    paths=dataset.all_photos_path[Path(dataset.all_sketches_path[index]).parent.name]
                    picked=paths[np.random.default_rng(sample_seed(args.seed,epoch,index)).integers(len(paths))]
                    pi=dataset.photo_path_to_index[picked];batch=dataset[(epoch,index)]
                    torch.testing.assert_close(batch[7],cache['photo']['attention_target'][pi])
                    torch.testing.assert_close(batch[5],apply_mask(batch[0],cache['photo']['attention_mask'][pi]))
                    torch.testing.assert_close(batch[8],cache['sketch']['attention_target'][index])
            # Both ablations reuse identical full teacher targets; no teacher rebuild.
            with patch('src.mask_guided_cache.load_teacher',side_effect=AssertionError('unexpected rebuild')):
                args.mask_strategy='random'
                repeated=prepare_mask_cache(args,dataset,metadata,root/'random_report')
                torch.testing.assert_close(repeated['photo']['full'],cache['photo']['full'])
                torch.testing.assert_close(dataset[0][8],cache['sketch']['random_target'][0])
                args.lambda_response=0
                prepare_mask_cache(args,dataset,metadata,root/'embedding_report')
                self.assertEqual(len(dataset[0]),5)
                args.mask_ratio=.5
                with self.assertRaisesRegex(ValueError,'provenance'):prepare_mask_cache(args,dataset,metadata,root/'bad')
                args.mask_ratio=.25
                Image.new('RGB',(8,8),'red').save(dataset.all_sketches_path[0])
                with self.assertRaisesRegex(ValueError,'provenance'):prepare_mask_cache(args,dataset,metadata,root/'bad')
            before=Path(args.mask_cache_path).read_bytes()
            with self.assertRaises(FileExistsError):save_cache(args.mask_cache_path,cache)
            self.assertEqual(Path(args.mask_cache_path).read_bytes(),before)
            self.assertFalse(list(root.glob('*.tmp')))


if __name__=='__main__':unittest.main()
