import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def main(filename):
    # 读取数据
    with open(filename, 'r') as file:
        data = [line.strip() for line in file.readlines()]

    # 将数据转化为Pandas DataFrame
    df = pd.DataFrame(data, columns=['data'])

    # 拆分数据集为训练集和其他
    train, other = train_test_split(df, test_size=0.02, random_state=42)

    # 拆分剩余数据为验证集和测试集
    val, test = train_test_split(other, test_size=0.5, random_state=42)

    # 保存拆分的数据集
    train.to_csv(filename.replace('.txt', '_train.txt'), index=False, header=False)
    val.to_csv(filename.replace('.txt', '_val.txt'), index=False, header=False)
    test.to_csv(filename.replace('.txt', '_test.txt'), index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into train, validation and test sets.')
    parser.add_argument('filename', type=str, help='File path to be split.')
    args = parser.parse_args()

    main(args.filename)