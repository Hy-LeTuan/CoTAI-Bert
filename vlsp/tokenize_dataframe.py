from read_and_process import read_and_return_df
from transformers import AutoTokenizer
from datasets import Dataset
from glob import glob
import os


dataset_type = "test"
text_files = sorted(glob(os.path.join("./data/formatted_data/2016/new", dataset_type, "**", "*.txt"), recursive=True))
tokenizer = AutoTokenizer.from_pretrained("../tokenizer/trained_tokenizer/tokenizer-50k/")
tokenizer.model_max_length = 10000000


def mlm_tokenize_dataset(row):
    word_list = row["words"]
    tokenizer_output = tokenizer(word_list, is_split_into_words=True, return_tensors="pt", truncation=False)

    return tokenizer_output


if __name__ == "__main__":
    df = read_and_return_df(text_files=text_files)
    ds = Dataset.from_spark(df, load_from_cache_file=False)

    ds = ds.map(mlm_tokenize_dataset, batched=False, num_proc=os.cpu_count())
    ds.save_to_disk(f"./data/mlm_data/2016/{dataset_type}")
