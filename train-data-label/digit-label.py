import unicodedata

# 初始化一个空的集合来存储非数字字符
non_digit_chars = set()

with open('./chineseocr/rec_digit_label.txt', 'r', encoding='utf-8') as file:
    data = file.read()


# 检查每个字符
for char in data:
    # 如果字符不是数字
    if not char.isdigit() and not unicodedata.category(char).startswith('C'):
        # 添加字符到集合
        non_digit_chars.add(char)

# 打印集合中的每个字符，每行一个
for char in non_digit_chars:
    print(char)
