from datasets import load_dataset


if __name__ == "__main__":
    train_dataset = load_dataset("arrow", data_files=["../data_all/data_ner/train/data_00000.arrow"], cache_dir=None, split="train")
    train_dataset_1 = load_dataset("arrow", data_files=["../data_all/data_ner/train/data_00001.arrow"], cache_dir=None, split="train")

    print(train_dataset)
    print("-----------------")
    print(train_dataset_1)

    input_id = train_dataset["input_ids"][0]
    input_id_1 = train_dataset_1["input_ids"][0]

    attention_mask = train_dataset["attention_mask"][0]
    attention_mask_1 = train_dataset_1["attention_mask"][0]

    token_type_ids = train_dataset["token_type_ids"][0]
    token_type_ids_1 = train_dataset_1["token_type_ids"][0]

    print(input_id)
    print(input_id_1)
    print("-----------------")

    print(attention_mask)
    print(attention_mask_1)
    print("-----------------")

    print(token_type_ids)
    print(token_type_ids_1)
    print("-----------------")
