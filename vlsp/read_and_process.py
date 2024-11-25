from pyspark.sql import SparkSession, Row
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StringType, ArrayType, StructField, StructType
from glob import glob
import os

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("OFF")


def read_format_and_return_df(text_files: list) -> DataFrame:
    def split_and_divide(row) -> Row:
        content = row["value"]
        word_list, tag_list = content.split("\t")

        # split each list into actual list
        word_list = word_list.split(" ")
        tag_list = tag_list.split(" ")

        return Row(words=word_list, tags=tag_list)

    df = spark.read.text(text_files)
    rdd = df.rdd.map(split_and_divide)

    schema = StructType([
        StructField("words", ArrayType(StringType()), True),
        StructField("tags", ArrayType(StringType()), True),
    ])

    # convert back to dataframe
    df = rdd.toDF(schema=schema)

    return df


def read_and_return_df(text_files: list) -> DataFrame:
    df = spark.read.text(text_files)
    df = df.dropna(how='any')  # Drop rows with any null values
    return df


if __name__ == "__main__":
    dataset_type = "train"
    text_files = sorted(glob(os.path.join("./data/formatted_data/2016/new", dataset_type, "**", "*.txt"), recursive=True))
    df = read_format_and_return_df(text_files)

    df.show()
