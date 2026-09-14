"""Small end-to-end fixture: real CLIP/Lightning, fake crop teacher; no downloads."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from clip.model import CLIP, convert_weights
from src.dataset import TrainDataset, TeacherFeatureDataset, WorkerInvariantSampler
from src.model import ZS_SBIR
from src.semantic_region_cache import prepare_region_cache, file_hash
from src.semantic_region_inference import load_region_checkpoint

torch.set_num_threads(4)


def arguments(root):
    tree=ast.parse(Path('src/train.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare))
    nodes=[]
    for node in main.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in node.targets):break
        nodes.append(node)
    scope={'argparse':argparse,'UNSEEN_CLASSES':{'sketchy_1':[]}}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'test_parser','exec'),scope)
    args=scope['parser'].parse_args(['--root',str(root),'--retrieval_head','semantic_region',
        '--lambda_domain','0','--lambda_modality','0','--teacher_pretrain_epochs','1'])
    args.max_size=32;args.workers=0;args.n_ctx_visual=3;args.prompt_depth=2
    args.region_cache_dir=str(root/'regions');args.teacher_cache_path=str(root/'teacher.pt')
    args.region_shard_size=2;args.region_calibration_per_class=3;args.region_diagnostic_interval=1
    names=('src/model.py','src/train.py','src/dataset.py','src/losses.py','src/teacher_prompts.py',
           'clip/model.py','src/semantic_region.py','src/semantic_region_cache.py','src/semantic_region_diagnostics.py')
    args.training_source_sha256={n:file_hash(n) for n in names}
    args.teacher_prompt_seed=42
    args.photo_text_kd_temperature=.15;args.sketch_text_kd_temperature=.02
    return args


def fixture(root):
    rng=np.random.default_rng(10)
    for modality in ('sketch','photo'):
        for category in ('seen_a','seen_b'):
            d=root/modality/category;d.mkdir(parents=True)
            for i in range(3):
                rgb=rng.integers(0,256,(32,32,3),dtype=np.uint8)
                Image.fromarray(rgb).save(d/('%d.png'%i))
    args=arguments(root)
    torch.manual_seed(9)
    clip=CLIP(512,32,2,64,8,77,49408,64,1,1).eval()
    convert_weights(clip)
    with patch('src.model._load_clip_model',return_value=clip),patch('src.model._load_teacher',return_value=None):
        module=ZS_SBIR(args,['seen_a','seen_b']).eval()
    dataset=TrainDataset(args)
    full={}
    with torch.no_grad():
        for modality,paths in [('sketch',dataset.all_sketches_path),('photo',dataset.all_photo_paths)]:
            images=torch.stack([TeacherFeatureDataset(paths,32)[i] for i in range(len(paths))])
            z=module.model.encode_student_image(images,modality).float()
            full[modality]=torch.cat((z,torch.zeros_like(z)),dim=-1).half()
    torch.save({'metadata':module.model._teacher_cache_metadata(dataset),
                'teacher_sketch_features':full['sketch'],'teacher_photo_features':full['photo'],
                'teacher_prompt_state_dict':{'dummy':torch.ones(1)}},args.teacher_cache_path)
    dataset.set_teacher_features(full['sketch'],full['photo'])
    return module,dataset


class Controller(nn.Module):
    def __init__(self,module):
        super().__init__();self.dummy=nn.Parameter(torch.ones(1))
        object.__setattr__(self,'student',module)
    def forward(self,images,modality):
        z=self.student.model.encode_student_image(images,modality).float()
        return torch.cat((z,torch.zeros_like(z)),dim=-1)


def prepare(module,dataset):
    teacher=SimpleNamespace(visual=SimpleNamespace(conv1=SimpleNamespace(weight=torch.ones(1,dtype=torch.float16))))
    matrix=torch.cat((torch.eye(512),torch.zeros(512,512)))
    with patch('open_clip.create_model',return_value=MockTeacher(teacher.visual)), \
         patch('src.teacher_prompts.build_teacher_prompt_controller',side_effect=lambda *a:Controller(module)), \
         patch('src.semantic_region_cache.fit_alignment',return_value=matrix):
        prepare_region_cache(module,dataset,None,None)


class MockTeacher(nn.Module):
    def __init__(self,visual):super().__init__();self.visual=visual


class IntegrationTests(unittest.TestCase):
    def test_sampler_uses_restored_lightning_epoch(self):
        sampler=WorkerInvariantSampler(list(range(8)),42)
        sampler.epoch_source=SimpleNamespace(current_epoch=7)
        self.assertTrue(all(epoch==7 for epoch,_ in sampler))
        sampler.epoch_source.current_epoch=8
        self.assertTrue(all(epoch==8 for epoch,_ in sampler))

    def test_prepare_resume_and_dataset_indexing(self):
        with TemporaryDirectory() as td:
            module,dataset=fixture(Path(td));prepare(module,dataset)
            directory=Path(module.args.region_cache_dir)
            hashes={p.name:file_hash(p) for p in directory.glob('*.pt')}
            with patch('open_clip.create_model',side_effect=AssertionError('Cache resume loaded teacher')):
                prepare_region_cache(module,dataset,None,None)
            self.assertEqual(hashes,{p.name:file_hash(p) for p in directory.glob('*.pt')})
            sample=dataset[(0,2)]
            self.assertEqual(len(sample),15)
            self.assertTrue(torch.equal(sample[6],dataset.region_targets['sketch']['crops'][2]))
            teacher_index=next(i for i,z in enumerate(dataset.teacher_photo_features) if torch.equal(z,sample[2]))
            self.assertTrue(torch.equal(sample[5],dataset.region_targets['photo']['crops'][teacher_index]))
            module.args.region_temperature*=2
            with self.assertRaisesRegex(ValueError,'configuration'):prepare_region_cache(module,dataset,None,None)

    def test_lightning_training_checkpoint_and_inference(self):
        with TemporaryDirectory() as td:
            root=Path(td);module,dataset=fixture(root);prepare(module,dataset)
            train=DataLoader(dataset,batch_size=3,shuffle=False,num_workers=0)
            valid=[]
            for modality,paths in [('sketch',dataset.all_sketches_path),('photo',dataset.all_photo_paths)]:
                images=TeacherFeatureDataset(paths,32)
                data=[(images[i],i//3) for i in range(len(images))]
                valid.append(DataLoader(data,batch_size=3))
            clip_before={n:p.detach().clone() for n,p in module.model.clip_model.named_parameters()}
            logger=CSVLogger(str(root/'logs'),name='region_fixture')
            trainer=Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',devices=1,max_epochs=1,logger=logger,
                            enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,
                            num_sanity_val_steps=0,log_every_n_steps=1)
            module.cpu()
            trainer.validate(module,dataloaders=valid,verbose=False)
            trainer.fit(module,train,valid)
            self.assertEqual(trainer.global_step,2)
            for n,p in module.model.clip_model.named_parameters():self.assertTrue(torch.equal(p.detach().cpu(),clip_before[n].cpu()))
            self.assertTrue(all(p.grad is None for p in module.model.photo_visual_prompt.parameters()))
            report=Path(logger.log_dir)/'region_diagnostics'
            for name in ('steps.csv','gradients.csv','epochs.csv','attention_epoch_00.png','training_diagnostics.png'):
                self.assertTrue((report/name).is_file(),name)
            import csv
            with (report/'epochs.csv').open(newline='',encoding='utf-8') as stream:epoch_rows=list(csv.DictReader(stream))
            self.assertEqual(epoch_rows[0]['photo_descriptor'],'')
            self.assertNotEqual(epoch_rows[-1]['photo_descriptor'],'')
            self.assertEqual(int(epoch_rows[-1]['global_step']),2)
            checkpoint=root/'final.ckpt';trainer.save_checkpoint(checkpoint)
            module.cpu()
            with patch('src.model._load_teacher',side_effect=AssertionError('Teacher used at inference')), \
                 patch('clip.clip.download_model',side_effect=AssertionError('Download used at inference')):
                loaded=load_region_checkpoint(checkpoint)
            x=next(iter(valid[0]))[0]
            with torch.no_grad():
                expected=module.model.extract_feature(x,'sketch')
                actual=loaded.model.extract_feature(x,'sketch')
            self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
            # Resume actual model, Adam moments and scheduler, then complete one more epoch.
            resumed=Trainer(accelerator='cpu',devices=1,max_epochs=2,logger=CSVLogger(str(root/'logs'),name='region_resume'),
                            enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,
                            num_sanity_val_steps=0,log_every_n_steps=1)
            for p in loaded.model.region_head.parameters():p.requires_grad_(True)
            resumed.fit(loaded,train,valid,ckpt_path=str(checkpoint))
            self.assertEqual(resumed.global_step,4)
            self.assertEqual(resumed.optimizers[0].state_dict()['state'][0]['step'].item(),4)
            saved=torch.load(checkpoint,weights_only=False)
            self.assertTrue(saved['optimizer_states'])
            self.assertEqual(saved['experiment_config']['region_target_metadata'],module.args.region_target_metadata)


if __name__=='__main__':unittest.main()
