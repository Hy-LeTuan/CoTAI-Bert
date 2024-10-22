import regex

expression = regex.compile(r"(?:link.*?)?https?:\/\/[\w\-\.\/\@\?\#\:\=\&\%]+")

comment = '-Bạn nào cần vay tiền gấp bấm vào link bên dưới rồi điền thông tin nha\n-Hoặc ib mình tư vấn\n -khoảng vay từ 10tr-80tr \n-thủ tục vay đơn giản\n\n👉link : https://shorten.asia/8VFGxTbN/name?=user'
complicate_comment = '-Bạn nào cần vay tiền gấp bấm vào link bên dưới rồi điền thông tin nha\n-Hoặc ib mình tư vấn\n -khoảng vay từ 10tr-80tr \n-thủ tục vay đơn giản\n\n👉 https://shorten.asia/8VFGxTbN/name?=user&age=20/#3/sale=20%/'

tagged_comment = "Đoàn Thị Yến thiệt mà.ng a thươg lun đẹp nhứt"
expression_tagged = regex.compile(r"^(?:\p{Lu}[\p{Ll}]*[\s])+")

complicate_res = regex.findall(expression, complicate_comment)
tagged_name = regex.findall(expression_tagged, tagged_comment)

print(tagged_name)

# for word in complicate_res:
#     print(word)
#     complicate_comment = complicate_comment.replace(word, '')


# print("final result---------")
# print(complicate_comment)
