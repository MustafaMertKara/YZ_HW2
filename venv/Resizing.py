from os import listdir
from os.path import isfile, join
import numpy
import cv2

inputPath = '.\\pictures'

onlyfiles = [f for f in listdir(inputPath) if isfile(join(inputPath, f))]

width1 = 350
height1 = 450
width2 = 768
height2 = 1024
width3 = 150
height3 = 270
width4 = 50
height4 = 50
width5 = 480
height5 = 640
dim1 = (width1, height1)
dim2 = (width2, height2)
dim3 = (width3, height3)
dim4 = (width4, height4)
dim5 = (width5, height5)


images = numpy.empty(len(onlyfiles), dtype=object)
resized1 = numpy.empty(len(onlyfiles), dtype=object)
resized2 = numpy.empty(len(onlyfiles), dtype=object)
resized3 = numpy.empty(len(onlyfiles), dtype=object)
resized4 = numpy.empty(len(onlyfiles), dtype=object)
resized5 = numpy.empty(len(onlyfiles), dtype=object)

for i in range(0, len(onlyfiles)):
    images[i] = cv2.imread(join(inputPath, onlyfiles[i]))
    resized1[i] = cv2.resize(images[i], dim1, interpolation=cv2.INTER_AREA)
    resized2[i] = cv2.resize(images[i], dim2, interpolation=cv2.INTER_AREA)
    resized3[i] = cv2.resize(images[i], dim3, interpolation=cv2.INTER_AREA)
    resized4[i] = cv2.resize(images[i], dim4, interpolation=cv2.INTER_AREA)
    resized5[i] = cv2.resize(images[i], dim5, interpolation=cv2.INTER_AREA)

    cv2.imwrite('.\\350x450\\{}.jpg'.format(i), resized1[i])
    cv2.imwrite('.\\768x1024\\{}.jpg'.format(i), resized2[i])
    cv2.imwrite('.\\150x270\\{}.jpg'.format(i), resized3[i])
    cv2.imwrite('.\\50x50\\{}.jpg'.format(i), resized4[i])
    cv2.imwrite('.\\480x640\\{}.jpg'.format(i), resized5[i])

cv2.imshow("Resized image", resized5[i])
print('Resized Dimensions : ', resized5[i].shape)


cv2.waitKey(0)
cv2.destroyAllWindows()