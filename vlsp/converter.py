from glob import glob


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


def read_and_join_list(text_file_paths, destination):
    for file_counter, filepath in enumerate(text_file_paths):
        with open(filepath, "r", encoding="utf-8") as input_file:
            output_file = open(
                f"./data/formatted_data/2016/new/{destination}/{str(file_counter).zfill(5)}.txt",
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


if __name__ == "__main__":
    # get all .txt file paths in NER 2016 dataset

    origin = ["./data/NER2016-TrainingData-3-3-2017-txt/**/*.txt",
              "./data/NER2016-TestData-16-9-2016/**/*.txt"]
    destination = ["train", "test"]

    for o, d in zip(origin, destination):
        text_file_paths = sorted(glob(o, recursive=True))
        # extract_and_reformat(text_file_paths, destination=d)
        read_and_join_list(text_file_paths=text_file_paths, destination=d)
