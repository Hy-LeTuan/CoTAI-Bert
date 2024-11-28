def tokenize_and_align(batch, tokenizer, label_map, tag_column_name="tags"):
    tokenized = tokenizer(batch["words"], is_split_into_words=True,
                          truncation=True, padding=True, return_tensors="pt")
    labels = []
    # align tokens
    for i, label in enumerate(batch[f"{tag_column_name}"]):
        if len(label) != len(batch["words"][i]):
            print(
                f"length does not match, label: {len(label)} || word: {len(batch['words'][i])}")
            print(f"label: {label} || word: {batch['words'][i]}")

        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_map[label[word_idx]])
            else:
                label_ids.append(label_map[label[word_idx]])

            previous_word_idx = word_idx

        labels.append(label_ids)

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "token_type_ids": tokenized["token_type_ids"],
        "labels": labels,
    }
