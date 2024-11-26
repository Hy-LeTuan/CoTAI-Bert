import sys
import os
sys.path.append(os.path.join(os.getcwd(), ".."))
from vlsp import read_and_process 
from glob import glob
from datasets import Dataset
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("../tokenizer/trained_tokenizer/tokenizer-50k")


def tokenize_dataset(batch):
    return tokenizer(batch["words"], is_split_into_words=True, truncation=True, padding=True, return_tensors="pt")


text_files = sorted(glob(os.path.join("../vlsp/data/formatted_data/2016", "**", "*.txt"), recursive=True))
df = read_and_process.read_format_and_return_df(text_files)
ds = Dataset.from_spark(df, cache_dir=None)

ds = ds.map(tokenize_dataset, batched=True)
print(ds)
