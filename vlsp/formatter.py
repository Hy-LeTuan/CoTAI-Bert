from glob import glob
import regex as re


# read file and extract tags
def extract_and_reformat(text_file_paths, destination):
    for file_counter, filepath in enumerate(text_file_paths):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.readline()
            output_file = open(
                f"./data/formatted_data/2016/{destination}/{str(file_counter).zfill(5)}.csv", "w", encoding="utf-8")

            # write header
            output_file.write("Word,Tag\n")

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


def read_and_join_list(text_file_paths, destination, year):
    for file_counter, filepath in enumerate(text_file_paths):
        with open(filepath, "r", encoding="utf-8") as input_file:
            output_file = open(
                f"./data/formatted_data/{year}/{destination}/{str(file_counter).zfill(5)}.txt",
                "w",
                encoding="utf-8"
            )

            content = input_file.readline()
            line_counter = 0

            while content:
                if line_counter < 3:
                    line_counter += 1
                    content = input_file.readline()
                else:
                    # strip trailing space
                    content = content.strip()

                    if content == "<s>":
                        content = input_file.readline()
                        content = content.strip()

                        # create word list and tag list
                        word_list = []
                        tag_list = []

                        while content.strip() != "</s>":
                            content = content.split("\t")
                            word = content[0]
                            tag = content[3]

                            # replace quotation with double quotations
                            word = word.replace('"', '""')

                            word_list.append(word)
                            tag_list.append(tag)

                            # continue reading file
                            content = input_file.readline()

                        # write output to a file
                        word_list_concat = " ".join(word_list)
                        tag_list_concat = " ".join(tag_list)

                        output_file.write(f"{word_list_concat}\t{tag_list_concat}\n")
                    else:
                        content = input_file.readline()


def extract_and_reformat_muc(text_file_paths, destination, year):
    re_pattern_with_tag = r"(<ENAMEX[^>]*>.*?<\/ENAMEX>)"
    for file_counter, filepath in enumerate(text_file_paths):
        with open(filepath, "r", encoding="utf-8") as input_file:
            output_file = open(
                f"./data/formatted_data/{year}/{destination}/{str(file_counter).zfill(5)}.txt",
                "w",
                encoding="utf-8"
            )

            line = input_file.readline()

            while line:
                regex_res_with_tag = re.findall(re_pattern_with_tag, line)

                if regex_res_with_tag:
                    for tag_phrase in regex_res_with_tag:
                        content = re.findall(r"<ENAMEX[^>]*>(.*?)<\/ENAMEX>", tag_phrase)[-1]

                        # replace the whole tag with only the content for MLM task
                        line = line.replace(tag_phrase, content)

                # write line with no tag and only words
                if len(line) > 0:
                    output_file.write(f"{line}")

                line = input_file.readline()


if __name__ == "__main__":
    year = 2018
    # origin = ["./data/NER2016-TrainingData-3-3-2017-txt/**/*.txt",
    #           "./data/NER2016-TestData-16-9-2016/**/*.txt"]
    origin = [
        "./data/VLSP2018-NER-dev/**/*.muc",
        "./data/VLSP2018-NER-train/VLSP2018-NER-train-Jan14/**/*.muc",
    ]
    destination = ["train", "test"]

    for o, d in zip(origin, destination):
        text_file_paths = sorted(glob(o, recursive=True))
        # extract_and_reformat(text_file_paths, destination=d, year=year)
        # read_and_join_list(text_file_paths=text_file_paths, destination=d)
        extract_and_reformat_muc(
            text_file_paths=text_file_paths,
            destination=d,
            year=year
        )
