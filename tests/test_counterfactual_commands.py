"""Validate the delivered Kaggle commands against the actual production parser."""
import argparse
import ast
from pathlib import Path
import re
import shlex
import sys
import unittest
from unittest.mock import patch


class KaggleCommandTests(unittest.TestCase):
    def test_all_delivered_commands_parse_without_downloads_or_training(self):
        project=Path(__file__).resolve().parents[1]
        train=project/'src/train.py'
        tree=ast.parse(train.read_text(encoding='utf-8'))
        main=next(n for n in tree.body if isinstance(n,ast.If) and '__main__' in ast.unparse(n.test))
        # Keep parser definitions/default resolution/source hashing. Stop before
        # logger, dataset construction, teacher downloads or GPU training.
        stop=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='logger' for t in n.targets))
        main.body=main.body[:stop]
        ast.fix_missing_locations(tree)
        code=compile(tree,str(train),'exec')
        paths=sorted((project/'test').glob('kaggle_avcrd_*.ipy'))
        self.assertGreaterEqual(len(paths),9)
        for path in paths:
            with self.subTest(command=path.name):
                text=path.read_text(encoding='utf-8')
                command=text.split('!python -m src.train',1)[1].replace('\\\n',' ')
                command=re.sub(r'--exp_name\s+[^\n]+?\$\(date[^)]+\)', '--exp_name command_test',command)
                arguments=shlex.split(command)
                scope={'__name__':'__main__','__file__':str(train)}
                with patch.object(sys,'argv',['src.train']+arguments):exec(code,scope)
                args=scope['args']
                self.assertEqual(args.batch_size,64);self.assertEqual(args.seed,42)
                self.assertEqual(args.backbone,'ViT-B/32')
                self.assertFalse(args.ckpt_path)
                self.assertIn('src/counterfactual_cache.py',args.training_source_sha256)
                if 'audit' in path.name:self.assertTrue(args.avcrd_prepare_only)
                if 'unverified' in path.name:self.assertEqual(args.avcrd_selection,'attention_first')

    def test_unsafe_or_ambiguous_training_configs_are_rejected_early(self):
        from src.counterfactual_cache import add_arguments,validate_arguments
        parser=argparse.ArgumentParser();add_arguments(parser)
        options=parser.parse_args(['--lambda_avcrd','1'])
        options.batch_size=64;options.teacher_pretrain_epochs=1;options.lambda_av=0;options.lambda_global_feature=0
        validate_arguments(parser,options)
        for name,value in [('avcrd_audit_per_class',2),('avcrd_teacher_batch_size',0),('lambda_av',1),('teacher_pretrain_epochs',0)]:
            with self.subTest(field=name):
                original=getattr(options,name);setattr(options,name,value)
                with self.assertRaises(SystemExit):validate_arguments(parser,options)
                setattr(options,name,original)

if __name__=='__main__':unittest.main()
