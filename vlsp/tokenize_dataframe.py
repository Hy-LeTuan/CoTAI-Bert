from read_and_process import read_and_return_df
from transformers import AutoTokenizer
from datasets import Dataset
from glob import glob
import os

dataset_type = "train"
text_files = sorted(glob(os.path.join("./data/formatted_data/2016/new", dataset_type, "**", "*.txt"), recursive=True))
tokenizer = AutoTokenizer.from_pretrained("../tokenizer/trained_tokenizer/tokenizer-50k/")


def mlm_tokenize_dataset(batch):
    word_list = batch["words"]
    tokenizer_output = tokenizer(word_list, is_split_into_words=True, return_tensors="pt")

    return tokenizer_output


if __name__ == "__main__":
    df = read_and_return_df(text_files=text_files)
    ds = Dataset.from_spark(df, load_from_cache_file=False)

    ds = ds.map(mlm_tokenize_dataset, batched=True, num_proc=os.cpu_count())
    ds.save_to_disk("./data/mlm_data/2016/train")
