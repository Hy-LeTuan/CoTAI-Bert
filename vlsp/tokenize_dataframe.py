from read_and_process import read_and_return_df
from transformers import AutoTokenizer
from glob import glob
import os

dataset_type = "train"
text_files = sorted(glob(os.path.join("./data/formatted_data/2016/new", dataset_type, "**", "*.txt"), recursive=True))
tokenizer = AutoTokenizer.from_pretrained("../tokenizer/trained_tokenizer/tokenizer-50k/")

def tokenize_and_align(row):
    word_list = row["words"]
    tag_list = row["tags"]

    tokenizer_output = tokenizer(word_list, is_split_into_words=True, return_tensors="pt")

    input_ids = tokenizer_output["input_ids"]
    attention_mask = tokenizer_output["attention_mask"]
    token_type_ids = tokenizer_output["token_type_ids"]

    print(input_ids)
    print(attention_mask)
    print(token_type_ids)

if __name__ == "__main__":
    df = read_and_return_df(text_files=text_files)
    word_list = df.collect()[0]["words"]
    print(word_list)
    print("--------")

