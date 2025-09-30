import torch
from torch.utils.data import DataLoader
from .dataset import InfiniteDataReader

def worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % (2**32)
    import random, numpy as np
    np.random.seed(base_seed); random.seed(base_seed); torch.manual_seed(base_seed)


def create_dataloader(batch_size: int, 
                      metas_path: str, 
                      num_actions: int,
                      training: bool,
                      action_mode: str,
                      rel_idx = []
                      ):
    return DataLoader(
        InfiniteDataReader(metas_path, num_actions=num_actions, training=training, action_mode = action_mode, rel_idx = rel_idx),
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )