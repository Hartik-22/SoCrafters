import cv2
import os
import time
import joblib
import numpy as np

# Load trained model
from intruder_pipeline import preprocess
model = joblib.load("intruder_model.pkl")

# HOG extractor (same as training)
def extract_hog(img):
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    return hog.compute(img).flatten()

# Choose folder to simulate camera
STREAM_FOLDER = "dataset/empty"   # change to empty to test

files = os.listdir(STREAM_FOLDER)

while True:
    for f in files:
        path = os.path.join(STREAM_FOLDER,f)

        frame = cv2.imread(path,0)
        frame = preprocess(frame)

        if frame is None:
            continue

        start = time.perf_counter()

        feat = extract_hog(frame).reshape(1,-1)
        score = model.decision_function(feat)[0]

        # safety margin
        THRESHOLD = 0.8

        if score > THRESHOLD:
            pred = 1
        else:
            pred = 0


        elapsed = time.perf_counter() - start
        fps = 1/elapsed if elapsed > 0 else 0


        frame = cv2.resize(frame,(256,256))

        if pred == 1:
            text="HUMAN DETECTED"
            color=255
        else:
            text="SAFE"
            color=255

        cv2.putText(frame,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        cv2.putText(frame,f"FPS:{fps:.2f}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
        print(text, " | FPS:", round(fps,2))
        time.sleep(0.2)

        # cv2.imshow("Intruder Detection",frame)

        # if cv2.waitKey(30)==27:
        #     break
