import tempfile
import unittest
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from src.train import save_final_checkpoint


class FinalCheckpointTests(unittest.TestCase):
    def test_final_weights_saved_when_monitored_metric_gets_worse(self):
        class Tiny(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def training_step(self, batch, batch_idx):
                return self.weight.square()

            def on_train_epoch_end(self):
                self.log('precision', 1. / (self.current_epoch + 1))

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=.1)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = ModelCheckpoint(dirpath=tmp, monitor='precision', mode='max',
                                         save_top_k=1, save_last=True)
            trainer = pl.Trainer(accelerator='cpu', max_epochs=3, logger=False,
                                 enable_progress_bar=False, enable_model_summary=False,
                                 callbacks=[checkpoint])
            model = Tiny()
            trainer.fit(model, DataLoader(TensorDataset(torch.ones(1)), batch_size=1))
            path = save_final_checkpoint(trainer, tmp)
            final = torch.load(path, map_location='cpu', weights_only=False)
            best = torch.load(checkpoint.best_model_path, map_location='cpu', weights_only=False)
            self.assertEqual(final['global_step'], 3)
            self.assertEqual(best['global_step'], 1)
            torch.testing.assert_close(final['state_dict']['weight'], model.weight)
            self.assertFalse(torch.equal(final['state_dict']['weight'], best['state_dict']['weight']))
            self.assertEqual(Path(path).name, 'final.ckpt')
