import regex

# expression = regex.compile(r"(?:link.*?)?https?:\/\/[\w\-\.\/\@\?\#\:\=\&\%]+")


# phone_number_expression = regex.compile(r"\+?0?\d{7,10}")
# comment_with_phone = "Tặng oil 0399695971"

# sentence_comment = "<person> ưng hàng ,chị cho em xin CÂN NẶNG + ĐỊA CHỈ + SĐT để đặt hàng nhá. Em sẽ kiểm tra lại size và soạn sẵng hàng hàng cho vừa đẹp với thông số size chị cung để gửi hàng cho chị nhé"
# sentence_pattern = regex.compile(r"(.*(?<=[\p{Ll}])\.\s+)")

# print(regex.split(sentence_pattern, sentence_comment))
# print(regex.findall(phone_number_expression, comment_with_phone))

# comment = '-Bạn nào cần vay tiền gấp bấm vào link bên dưới rồi điền thông tin nha\n-Hoặc ib mình tư vấn\n -khoảng vay từ 10tr-80tr \n-thủ tục vay đơn giản\n\n👉link : https://shorten.asia/8VFGxTbN/name?=user'
# complicate_comment = '-Bạn nào cần vay tiền gấp bấm vào link bên dưới rồi điền thông tin nha\n-Hoặc ib mình tư vấn\n -khoảng vay từ 10tr-80tr \n-thủ tục vay đơn giản\n\n👉 https://shorten.asia/8VFGxTbN/name?=user&age=20/#3/sale=20%/'

# illegal_content = regex.compile(r"^[\?\.\#\@\-\+\=\$\^\&,]+$")
# print(regex.findall(illegal_content, ".a,....."))

tagged_comment = "Đinh Đức Mạnh có😌"
expression_tagged = regex.compile(
    r"^[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s[A-ZÀ-Ỹ][a-zà-ỹ]+)*")

# complicate_res = regex.findall(expression, complicate_comment)
tagged_name = regex.findall(expression_tagged, tagged_comment)

print(tagged_name)
# print("after removing---------")

# for word in tagged_name:
#     tagged_name = tagged_comment.replace(word, '')

# print(tagged_name)


# for word in complicate_res:
#     print(word)
#     complicate_comment = complicate_comment.replace(word, '')


# print("final result---------")
# print(complicate_comment)
