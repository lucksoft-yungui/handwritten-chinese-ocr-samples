from utils.dataset import ImageDataset
import numpy as np

import os,sys                                     

sys.path.append(os.getcwd()) 

def test_get_item():
    dataset = ImageDataset(
        data_path='/Users/peiyandong/Documents/code/ai/hw_train_data',
        data_label_path='./train-data-label/chineseocr',
        data_file_name='rec_digit_label',
        phase="train",
        img_shape=(1,128)
    )
    
    print(f"dataset length：{len(dataset)}")

    img,label = dataset[0]

    assert isinstance(label, str), "对象不是字符类型"
    assert isinstance(img, np.ndarray), "对象不是NumPy数组类型"
