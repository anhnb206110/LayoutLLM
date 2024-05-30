from datasets import load_dataset, load_from_disk
import os
def get_train_dataset(path):
    dataset = load_from_disk(path + "/train")
    dataset.set_format("torch")
    dataset = dataset.filter(lambda x: x['input_ids'].shape[1] + 197 <= 512)
    return dataset

def get_val_dataset(path):
    dataset = load_from_disk(path + "/val")
    dataset.set_format("torch")
    dataset = dataset.filter(lambda x: x['input_ids'].shape[1] + 197 <= 512)
    return dataset