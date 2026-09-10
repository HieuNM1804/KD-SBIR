"""Student-only P x K sampler; teacher pretraining retains main's sampler."""
from pathlib import Path
import torch
from src.dataset import sample_seed


class ClassBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, per_class, seed):
        if per_class < 1 or batch_size % per_class or batch_size//per_class < 2:
            raise ValueError("Class batches require at least two classes and integral P x K.")
        self.dataset, self.batch_size, self.per_class, self.seed = dataset, batch_size, per_class, seed
        self.epoch = 0
        self.classes = batch_size//per_class
        self.sketches = {c: [] for c in dataset.all_categories}
        self.photos = {c: [] for c in dataset.all_categories}
        for i, path in enumerate(dataset.all_sketches_path):
            self.sketches[Path(path).parent.name].append(i)
        for i, path in enumerate(dataset.all_photo_paths):
            self.photos[Path(path).parent.name].append(i)
        if len(self.sketches) < self.classes:
            raise ValueError("Not enough seen classes for requested P x K batch.")
        if any(len(v) < per_class for v in list(self.sketches.values())+list(self.photos.values())):
            raise ValueError("Each seen class must have K unique sketches and photos.")
        if len(dataset)//batch_size == 0:
            raise ValueError("No complete class batch.")

    def __len__(self):
        return len(self.dataset)//self.batch_size

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        g = torch.Generator().manual_seed(sample_seed(self.seed, epoch, 817))
        names = self.dataset.all_categories
        for _ in range(len(self)):
            chosen = torch.randperm(len(names), generator=g)[:self.classes].tolist()
            batch = []
            for index in chosen:
                category = names[index]
                sk, ph = self.sketches[category], self.photos[category]
                si = torch.randperm(len(sk), generator=g)[:self.per_class].tolist()
                pi = torch.randperm(len(ph), generator=g)[:self.per_class].tolist()
                batch.extend((epoch, sk[s], ph[p]) for s, p in zip(si, pi))
            yield batch
