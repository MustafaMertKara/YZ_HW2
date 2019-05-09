import cv2 as cv
import tensorflow as tf
import numpy as np
import os

address="YZ_Data/3/"


img=cv.imread(address+"IMAG0372.jpg",0)
print(os.path.isfile(address+"IMAG0372.jpg"))

img=cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

img=img.flatten()
