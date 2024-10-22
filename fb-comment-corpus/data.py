import json
import pandas as pd
import os
import regex
import time


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
        comment = comment.replace(word, "<link>")

    return comment


def clean_person(comment):
    expression = regex.compile(r"^(?:\p{Lu}[\p{Ll}]*[\s])+")
    matches = regex.findall(expression, comment)

    for word in matches:
        comment = comment.replace(word, "<person> ")

    return comment


def clean_data():
    root_dir = "./json_data"
    output_dir = "./json_data_cleaned"
    output_content = {
        "id": [],
        "content": []
    }
    for filename in os.listdir(root_dir):
        json_file = open(os.path.join(root_dir, filename),
                         "r", encoding="utf-8")

        object_list = json.load(json_file)

        for object in object_list:
            content = object["content"]

            content = clean_person(content)
            content = clean_url(content)

            print(content)
            time.sleep(0.5)

        json_file.close()

        break


if __name__ == "__main__":
    clean_data()
