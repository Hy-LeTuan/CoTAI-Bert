import pyarrow as pa
import pyarrow.feather as feather
import tqdm as tqdm


def convert_to_arrow(dataset, output_dir, numbering_digits=5, type: str = "train"):
    """
    dataset: type of IterableDatasetDict
    """

    content = []
    output_file_counter = 0

    for i, row in tqdm(enumerate(dataset[type])):
        content.append(row["text"])

        if i % 500000 == 0 and i != 0:
            table = pa.Table.from_arrays([pa.array(content)], names=["text"])
            output_filepath = f"{output_dir}/{str(output_file_counter).zfill(numbering_digits)}.arrow"

            with pa.output_stream(output_filepath) as stream:
                writer = pa.RecordBatchStreamWriter(stream, table.schema)
                writer.write_table(table)
                writer.close()

            # feather.write_feather(table, output_filepath , version=1)

            output_file_counter += 1
            content = []
