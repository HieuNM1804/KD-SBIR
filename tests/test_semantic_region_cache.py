from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import torch
from torch.nn import functional as F
from src.semantic_region_cache import atomic_save, atomic_json, valid_file, record_file, validate_shard


class CacheTests(unittest.TestCase):
    def test_atomic_save_cleans_failed_partial_and_preserves_existing(self):
        with TemporaryDirectory() as td:
            p=Path(td)/'data.pt';atomic_save(p,{'x':torch.ones(3)})
            original=p.read_bytes()
            def failing_save(value,tmp):
                Path(tmp).write_bytes(b'partial');raise RuntimeError('disk write failed')
            with patch('torch.save',side_effect=failing_save),self.assertRaises(RuntimeError):
                atomic_save(p,{'x':torch.zeros(3)})
            self.assertEqual(p.read_bytes(),original)
            self.assertFalse(p.with_name(p.name+'.tmp').exists())

    def test_verified_shards_detect_corruption(self):
        with TemporaryDirectory() as td:
            directory=Path(td);manifest={'files':{}}
            atomic_save(directory/'data.pt',{'x':torch.ones(3)})
            record_file(directory,manifest,'data.pt')
            self.assertTrue(valid_file(directory,manifest,'data.pt'))
            (directory/'data.pt').write_bytes(b'corrupt')
            self.assertFalse(valid_file(directory,manifest,'data.pt'))

    def test_shard_validation(self):
        data={'crops':F.normalize(torch.randn(3,4,8),dim=-1).half(),
              **{key:torch.full((3,4),.25) for key in ('prior','semantic','random')}}
        validate_shard(data,3,4,8)
        data['semantic'][0,0]=2
        with self.assertRaisesRegex(ValueError,'distribution'):validate_shard(data,3,4,8)


if __name__=='__main__':unittest.main()
