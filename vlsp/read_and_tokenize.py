from pyspark.sql import SparkSession, Row
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StringType, ArrayType, StructField, StructType
from datetime import datetime, date
import pandas as pd

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("OFF")


def split_and_divide(row) -> Row:
    content = row["value"]
    word_list, tag_list = content.split("\t")

    # split each list into actual list
    word_list = word_list.split(" ")
    tag_list = tag_list.split(" ")

    return Row(words=word_list, tags=tag_list)


def read_and_return_df(text_files: list) -> DataFrame:
    df = spark.read.text(text_files)

    # map the split function
    rdd = df.rdd.map(split_and_divide)
    schema = StructType([
        StructField("words", ArrayType(StringType()), True),
        StructField("tags", ArrayType(StringType()), True),
    ])

    # convert back to dataframe
    df = rdd.toDF(schema=schema)

    return df


if __name__ == "__main__":

    text_files = ["/home/hyle/Documents/vscode/NLPDataCollection/NLPDataCollection/vlsp/data/formatted_data/2016/new/train/00000.txt",]

    df = read_and_return_df(text_files)

    row = df.collect()[0]

    print(row["words"])
    print("----")
    print(row["tags"])
