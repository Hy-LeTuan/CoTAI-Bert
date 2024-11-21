import pyarrow as pa
import pyarrow.feather as feather
import tqdm as tqdm
from hashlib import sha256
import time


def process_and_join(batch, num_tokens_file, tokenizer):
    content = []
    token_length = 0

    hashed = sha256(str(time.time()).encode("utf-8")).hexdigest()
    stream = pa.CompressedOutputStream(
        f"./temp/{hashed}.arrow", compression="gzip")
    outputs = tokenizer.batch_encode_plus(batch["text"])["input_ids"]

    for row in outputs:
        # remove [SEP] token
        row = row[:-1]

        if len(row) <= 128:
            content.append(row)
            token_length += len(row)
        else:
            for start_idx in range(0, len(row), 90):
                token_length += 128
                content.append(row[start_idx:start_idx+128])

    table = pa.Table.from_arrays(
        [pa.array(content)], names=["input_ids"])

    writer = pa.RecordBatchStreamWriter(stream, table.schema)
    writer.write_table(table)

    num_tokens_file.write(f"{token_length}\n")

    return batch


def convert_to_tokens(batch, tokenizer):
    outputs = tokenizer.batch_encode_plus(batch["text"])["input_ids"]
    return {"input_ids": outputs}


def convert_to_arrow(dataset, output_dir, max_row_per_line, prefix_digit=5, type: str = "train", text_column="text"):
    """
    dataset: type of IterableDatasetDict
    """

    content = []
    output_file_counter = 0

    for i, row in tqdm(enumerate(dataset[type])):
        content.append(row[text_column])

        if i % max_row_per_line == 0 and i != 0:
            table = pa.Table.from_arrays(
                [pa.array(content)], names=[text_column])
            output_filepath = f"{output_dir}/data_{str(output_file_counter).zfill(prefix_digit)}.arrow"

            with pa.CompressedOutputStream(output_filepath, compression="gzip") as stream:
                writer = pa.RecordBatchStreamWriter(stream, table.schema)
                writer.write_table(table)
                writer.close()

            output_file_counter += 1
            content = []


def convert_content_and_write_to_arrow(content, output_filepath, text_column="text", compression="gzip"):
    table = pa.Table.from_arrays(
        [pa.array(content)], names=[text_column])

    with pa.CompressedOutputStream(output_filepath, compression=compression) as stream:
        writer = pa.RecordBatchStreamWriter(stream, table.schema)
        writer.write_table(table)
        writer.close()
