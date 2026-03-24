import polars as pl
from torch.utils.data import DataLoader

from verb_overgen.train.train import TrainingSession
from verb_overgen.train.lib.dataloader_pychildes import DataBatchPyChildes
from verb_overgen.train.lib.utils import load_train_config

@__import__("ipdb").iex
def main():
    config = load_train_config()
    session = TrainingSession(config)
    _ = session.setup_tokenizer()
    token_id_pad = session.token_id_pad 
    _ = session.setup_datasets() 
    dataloader_train: DataLoader[DataBatchPyChildes] = session.setup_dataloaders()["train"]
    
    list_batch_info = []
    for index_batch, batch in enumerate(dataloader_train):            
        list_batch_info.append({
            "index_batch": index_batch,
            "datapoint_removed": batch.get("count_removed", 0),
        })
        
    df_batch_info = pl.DataFrame(list_batch_info)
    __import__("IPython").embed(using=False)

if __name__ == "__main__":
    main()
