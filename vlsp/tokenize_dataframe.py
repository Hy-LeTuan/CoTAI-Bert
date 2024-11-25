from read_and_process import read_format_and_return_df, read_and_return_df
from transformers import AutoTokenizer
from datasets import Dataset
from glob import glob
import os


tokenizer = AutoTokenizer.from_pretrained("../tokenizer/trained_tokenizer/tokenizer-50k/")
tokenizer.model_max_length = 1000000


def mlm_tokenize_dataset(sample, column_name="words", is_split_into_words=True):
    word_list = sample[column_name]
    tokenizer_output = tokenizer(
        word_list,
        is_split_into_words=is_split_into_words,
        truncation=False,
        return_tensors="pt"
    )

    tokenizer_output["input_ids"] = tokenizer_output["input_ids"][0]
    tokenizer_output["attention_mask"] = tokenizer_output["attention_mask"][0]
    tokenizer_output["token_type_ids"] = tokenizer_output["token_type_ids"][0]

    return tokenizer_output


def tokenize_for_2016():
    dataset_type_list = ["test", "train"]
    year = "2016"

    for dataset_type in dataset_type_list:
        text_files = sorted(glob(os.path.join("./data/formatted_data/", year, dataset_type, "**", "*.txt"), recursive=True))
        df = read_format_and_return_df(text_files=text_files)

        ds = Dataset.from_spark(df, load_from_cache_file=False, cache_dir=None, split="train")

        ds = ds.map(
            mlm_tokenize_dataset,
            batched=False,
            num_proc=os.cpu_count(),
            fn_kwargs={
                "column_name": "words",
                "is_split_into_words": True
            }
        )

        ds = ds.remove_columns(["words", "tags"])
        print(ds["input_ids"][0])
        print("--------")
        ds.save_to_disk(f"./data/mlm_data/{year}/{dataset_type}")


def tokenize_for_2018():
    dataset_type_list = ["test", "train"]
    year = "2018"

    for dataset_type in dataset_type_list:
        text_files = sorted(glob(os.path.join("./data/formatted_data/", year, dataset_type, "*.txt"), recursive=True))

        df = read_and_return_df(text_files=text_files)
        ds = Dataset.from_spark(df, load_from_cache_file=False, cache_dir=None, split="train")

        ds = ds.map(
            mlm_tokenize_dataset,
            batched=False,
            num_proc=os.cpu_count(),
            fn_kwargs={
                "column_name": "value",
                "is_split_into_words": False
            }
        )

        ds = ds.remove_columns(["value"])
        print(ds["input_ids"][0])
        print("--------")
        ds.save_to_disk(f"./data/mlm_data/{year}/{dataset_type}")


if __name__ == "__main__":
    tokenize_for_2016()
    tokenize_for_2018()
