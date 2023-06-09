
import os
import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

img_c=1
img_h=128

def imshow(tensor):
        # Convert tensor to numpy array
        image = tensor.numpy()

        # Transpose the dimensions from CxHxW to HxWxC
        image = np.transpose(image, (1, 2, 0))

        # Convert the range from [-1, 1] to [0, 1]
        image = image * 0.5 + 0.5

        # Display the image
        plt.imshow(image)
        plt.show()

def pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # plt.imshow(img)
            # plt.show()

            img = np.array(img)

            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
            
            # if the image has alpha channel, remove it
            if img.shape[2] == 4:
                img = img[:, :, :3]

            height, width = img.shape[:2]
            ratio = img_h/height
            new_width = int(width * ratio)
            img_resize = cv2.resize(img, (new_width, img_h), interpolation=cv2.INTER_AREA)

            # Ensure that grayscale images always have three dimensions
            if img_resize.ndim == 2:
                img_resize = img_resize[:, :, np.newaxis]

            # If we want a grayscale image but the image has three channels, convert it to grayscale
            if img_c == 1 and img_resize.shape[2] == 3:
                img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
                img_resize = img_resize[:, :, np.newaxis]

            # plt.imshow(img_resize)
            # plt.show()

            return img_resize
        
# to be removed
class ZerosPAD(object):
    def __init__(self, max_size):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size

    def __call__(self, img):
        img = self.toTensor(img)
        c, h, w = img.shape
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad

        return Pad_img


class NormalizePAD(object):
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size

    def __call__(self, img):      
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = \
                img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img
    

class AlignCollate(object):
    def __init__(self, imgH=48, PAD='ZerosPAD'):
        self.imgH = imgH
        self.PAD = PAD

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        maxW = 0
        for image in images:
            h, w, c = image.shape
            if w > maxW:
                maxW = w

        if self.PAD == 'ZerosPAD':
            trans = ZerosPAD((1, self.imgH, maxW))
        elif self.PAD == 'NormalizePAD':
            trans = NormalizePAD((1, self.imgH, maxW))
        else:
            raise ValueError("not expected padding.")

        padded_images = []
        for image in images:
            h, w, c = image.shape
            padded_images.append(trans(image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in padded_images], 0)

        return image_tensors, labels
    


imgArr = [(pil_loader(p),n) for p,n in [('/Users/peiyandong/Documents/code/ai/handwritten-chinese-ocr-samples/img-process/396.jpg','396'),
                                  ('/Users/peiyandong/Documents/code/ai/handwritten-chinese-ocr-samples/img-process/416.jpg','416'),
                                  ('/Users/peiyandong/Documents/code/ai/handwritten-chinese-ocr-samples/img-process/418.jpg','418'),
                                  ('/Users/peiyandong/Documents/code/ai/handwritten-chinese-ocr-samples/img-process/438.jpg','438'),
                                  ('/Users/peiyandong/Documents/code/ai/handwritten-chinese-ocr-samples/img-process/450.jpg','450')]]

ac = AlignCollate(imgH=128,PAD='NormalizePAD')
acArr = ac(imgArr)

_imgs,_names = acArr

for _img in _imgs:
     imshow(_img)
