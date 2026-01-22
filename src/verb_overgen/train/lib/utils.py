from lightning.pytorch.callbacks import Callback
import torch
import gc

class CacheCleanCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        gc.collect() 
        torch.cuda.empty_cache() 

