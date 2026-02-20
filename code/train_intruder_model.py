import cv2
import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def extract_hog(img):
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    return hog.compute(img).flatten()

X=[]
y=[]

for label,folder in enumerate(["empty","person"]):
    path="dataset/"+folder
    for file in os.listdir(path):
        img=cv2.imread(os.path.join(path,file),0)
        feat=extract_hog(img)
        X.append(feat)
        y.append(label)

X=np.array(X)
y=np.array(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearSVC()
model.fit(X_train,y_train)

pred=model.predict(X_test)
print(classification_report(y_test,pred))

joblib.dump(model,"intruder_model.pkl")
print("Model saved as intruder_model.pkl")

