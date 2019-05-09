from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import numpy as np
import csv
import cv2 as cv

DataAdress="YZ_Data/"

Size="768x1024"

imgs=np.load(DataAdress+Size+"_data.npy")
labels=np.load(DataAdress+Size+"_labels.npy")

mc=svm.SVC(kernel="poly",C=100000,degree=8,gamma="auto")
# mc= LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=500)

cv=StratifiedKFold(5,True)
# Butun skorların toplandığı list
EndScore=[]
for i in range(2):
    scores=cross_val_score(mc,imgs,labels,cv=cv)
    print(scores)
    EndScore.extend(scores)
print(EndScore)
print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(EndScore), np.std(EndScore) * 2))