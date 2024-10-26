import os
import regex as regex
from tqdm import tqdm


def clean_general(comment, ignore_newline=True):
    illegal_content = regex.compile(r"^[\?\.\#\@\-\+\=\$\^\&,]+$")
    if comment == "" or (ignore_newline == True and comment == "\n"):
        return "<empty>"
    elif len(regex.findall(illegal_content, comment)) != 0:
        return "<illegal>"
    else:
        return comment


def clean_url(comment):
    expression = regex.compile(
        r"(?:link.*?)?https?:\/\/[\w\-\.\/\@\?\#\:\=\&\%]+")
    matches = regex.findall(expression, comment)

    for word in matches:
        comment = comment.replace(word, "<link> ")

    return comment


output_file_counter = 0
output_dir = "./split_data"
max_line_per_file = 10000


def get_output_filename(file_counter) -> str:
    filename = f"./split_data/text_{str(file_counter).zfill(5)}.txt"
    return filename


with open("./data_vi.txt", "r", encoding="utf-8") as f:
    text = f.readlines()
    output_file = open(get_output_filename(
        output_file_counter), "w", encoding="utf-8")

    for i, line in tqdm(enumerate(text), total=len(text)):
        line = clean_url(line)
        line = clean_general(line)

        if line != "<illegal>" and line != "<empty>":
            output_file.write(line)
            if i != 0 and i % max_line_per_file == 0:
                output_file_counter += 1

                # open new file for writing
                output_file = open(get_output_filename(
                    output_file_counter), "w", encoding="utf-8")
