import torch
from glob import glob
import os


# read file and extract tags


def extract_and_reformat(text_file_paths, destination):
    for file_counter, filepath in enumerate(text_file_paths):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.readline()
            output_file = open(
                f"./data/formatted_data/2016/{destination}/{str(file_counter).zfill(5)}.csv", "w", encoding="utf-8")

            # write header
            output_file.write(f"Word,Tag")

            line_counter = 0
            while content:
                if line_counter <= 3:
                    line_counter += 1
                    content = f.readline()
                    continue
                # get tags
                tags = content.split("\t")

                # if tag is not a word
                if tags[0] == '<s>' or tags[0] == "</s>":
                    tags = []
                else:
                    tags[-1] = tags[-1].strip()

                if tags and len(tags) >= 4:
                    words = tags[0]
                    ner_tag = tags[3]
                    output_file.write(f"{words},{ner_tag}\n")

                content = f.readline()

            # prepare for next increment
            output_file.close()


if __name__ == "__main__":
    # get all .txt file paths in NER 2016 dataset

    origin = ["./data/NER2016-TrainingData-3-3-2017-txt/**/*.txt",
              "./data/NER2016-TestData-16-9-2016/**/*.txt"]
    destination = ["train", "test"]

    for o, d in zip(origin, destination):
        text_file_paths = sorted(glob(o, recursive=True))
        extract_and_reformat(text_file_paths, destination=d)
