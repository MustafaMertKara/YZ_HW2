import cv2 as cv
import tensorflow as tf
import numpy as np
import os

address="YZ_Data"


img=cv.imread(address+"IMAG0372.jpg",0)
print(os.path.isfile(address+"IMAG0372.jpg"))

img=cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

img=img.flatten()

for r , d ,f in os.walk(address):
    if r==address:
        for direc in d:
            n_address=os.path.join(address,direc)
            for r1,d1,f1 in os.walk(n_address):
                
