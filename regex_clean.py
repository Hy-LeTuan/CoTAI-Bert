import regex as regex


class RegexCleaner:
    def __init__(self):
        pass

    def clean_url(self, comment):
        expression = regex.compile(
            r"(?:link.*?)?https?:\/\/[\w\-\.\/\@\?\#\:\=\&\%]+")
        matches = regex.findall(expression, comment)

        for word in matches:
            comment = comment.replace(word, "<link> ")

        return comment

    def clean_person(self, comment):
        expression = regex.compile(r"^(?:\p{Lu}[\p{Ll}]*[\W]*?[\s]){2,}")

        matches = regex.findall(expression, comment)

        for word in matches:
            comment = comment.replace(word, "<person> ")

        return comment

    def clean_phone_number(self, comment):
        phone_number_expression = regex.compile(r"\+?0?\d{7,10}")
        matches = regex.findall(phone_number_expression, comment)

        for word in matches:
            comment = comment.replace(word, "")

        return comment

    def clean_general(self, comment, ignore_newline=True):
        illegal_content = regex.compile(r"^[\?\.\#\@\-\+\=\$\^\&,]+$")
        if comment == "" or (ignore_newline == True and comment == "\n"):
            return "<empty>"
        elif len(regex.findall(illegal_content, comment)) != 0:
            return "<illegal>"
        else:
            return comment

    def clean_trivial_sentences(self, comment):
        words = comment.split(" ")
        if len(words) < 2:
            return "<empty>"
        else:
            return comment

    def split_sentence(self, comment):
        """
        split sentences by punctuations. 
        """
        expression = r'(?<=[.!?])\s+'
        splitted = regex.split(expression, comment)

        res = []

        for sentence in splitted[0:-8:1]:
            if sentence != "":
                sentence = sentence.strip()
                sentence = self.clean_url(sentence)
                sentence = self.clean_general(sentence)

                if sentence == "<illegal>" or sentence == "<empty>":
                    continue

                res.append(sentence)

        return res
