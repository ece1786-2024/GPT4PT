from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset, Dataset, load_from_disk
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import BertTokenizerFast, BertTokenizer, GPT2Tokenizer, LlamaTokenizer, AutoTokenizer
from data_preprocessing import MBTIDataset

def get_dataloader(dataset:MBTIDataset, dataloader_drop_last:bool=True, shuffle:bool=False,
                   batch_size:int=16, dataloader_num_workers:int=0, dataloader_pin_memory:bool=True) -> DataLoader:

    # dataset is a collection of encodings and labels 
    # but the getitem is showing 'input_ids', 'attention_mask' and 'labels' which should be fine 
    dataloader = DataLoader(
                    dataset,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    # collate_fn=data_collator,
                    # drop_last=dataloader_drop_last,
                    num_workers=dataloader_num_workers,
                    pin_memory=False,
    )
    
    return dataloader