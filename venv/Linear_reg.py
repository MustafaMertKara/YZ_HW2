from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import numpy as np
import csv
import cv2 as cv

DataAdress="YZ_Data/"

Size="50x50"

imgs=np.load(DataAdress+Size+"_data.npy")
labels=np.load(DataAdress+Size+"_labels.npy")

# mc=svm.SVC(kernel="poly",C=100000,degree=8,gamma="auto")
mc= LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=1000)

cv=StratifiedKFold(10,True)
scores=cross_val_score(mc,imgs,labels,cv=cv)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))