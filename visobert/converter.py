import pyarrow as pa
import pyarrow.feather as feather
import tqdm as tqdm


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
