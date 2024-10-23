import json
import pandas as pd
import os
import regex
import time
from tqdm import tqdm


def process_original_comments():
    with open("./fb_comment_10m.jsonl", "r", encoding="utf-8") as f:
        max_object = 1000
        line_counter = 0
        file_counter = 0

        current_writing_file = f"json_data/fb_comment_text_{file_counter}.json"
        fw = open(current_writing_file, "w", encoding="utf-8")
        fw.write(f"[\n")

        content = f.readline()
        while content:
            json_content = json.loads(content)
            content = f.readline()
            if line_counter % max_object == 0 and line_counter != 0:
                fw.write(f"{json.dumps(json_content, indent=4)}\n")
                fw.write("]")
                fw.close()

                file_counter += 1
                current_writing_file = f"""json_data/fb_comment_text_{
                    file_counter}.json"""

                fw = open(current_writing_file, "w", encoding="utf-8")
                fw.write(f"[\n")

                line_counter += 1
            else:
                if content:
                    fw.write(f"{json.dumps(json_content, indent=4)},\n")
                else:
                    fw.write(f"{json.dumps(json_content, indent=4)}\n")
                line_counter += 1

        fw.write("]")
        fw.close()


def clean_url(comment):
    expression = regex.compile(
        r"(?:link.*?)?https?:\/\/[\w\-\.\/\@\?\#\:\=\&\%]+")
    matches = regex.findall(expression, comment)

    for word in matches:
        comment = comment.replace(word, "<link> ")

    return comment


def clean_person(comment):
    expression = regex.compile(r"^(?:\p{Lu}[\p{Ll}]*[\W]*?[\s]){2,}")

    matches = regex.findall(expression, comment)

    for word in matches:
        comment = comment.replace(word, "<person> ")

    return comment


def clean_non_word(comment):
    pass


def clean_phone_number(comment):
    phone_number_expression = regex.compile(r"\+?0?\d{7,10}")
    matches = regex.findall(phone_number_expression, comment)

    for word in matches:
        comment = comment.replace(word, "")

    return comment


def clean_general(comment):
    illegal_content = regex.compile(r"^[\?\.\#\@\-\+\=\$\^\&,]+$")
    if comment == "" or comment == "\n":
        return "<empty>"
    elif len(regex.findall(illegal_content, comment)) != 0:
        return "<illegal>"
    else:
        return comment


def clean_data():
    root_dir = "./json_data"
    output_dir = "./data_cleaned"

    for x, filename in tqdm(enumerate(os.listdir(root_dir))):
        json_file = open(os.path.join(root_dir, filename),
                         "r", encoding="utf-8")

        x = str(x).zfill(5)
        output_file = open(os.path.join(
            output_dir, f"fb_{x}.txt"), "w", encoding="utf-8")

        # get all records in json file
        object_list = json.load(json_file)

        for i, object in enumerate(object_list):
            content = object["content"]

            content = clean_url(content)
            content = clean_phone_number(content)
            content = clean_person(content)

            content = clean_general(content)

            if content != "<empty>" and content != "<illegal>":
                output_file.write(f"{content}\n")

        json_file.close()


if __name__ == "__main__":
    clean_data()
