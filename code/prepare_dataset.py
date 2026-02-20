import os
import shutil
import cv2
import numpy as np
from intruder_pipeline import preprocess
IMG_DIR = "images"
LBL_DIR = "labels"

OUT_PERSON = "dataset/person"
OUT_EMPTY = "dataset/empty"

os.makedirs(OUT_PERSON, exist_ok=True)
os.makedirs(OUT_EMPTY, exist_ok=True)

for img_name in os.listdir(IMG_DIR):

    if not img_name.lower().endswith(('.png','.jpg','.jpeg')):
        continue

    img_path = os.path.join(IMG_DIR, img_name)

    # corresponding label file
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(LBL_DIR, label_name)

    # check if label exists and has content
    has_person = False
    if os.path.exists(label_path):
        with open(label_path,'r') as f:
            content = f.read().strip()
            if content != "":
                has_person = True

    # read image
    img = cv2.imread(img_path)
    if img is None:
        continue

    # convert to grayscale 64x64 (VERY IMPORTANT)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = preprocess(gray)


    # SAME preprocessing as runtime
    blur = cv2.GaussianBlur(processed,(5,5),0)
    sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)
    mag = cv2.magnitude(sobelx,sobely)
    processed = np.uint8(np.clip(mag,0,255))

    # save
    if has_person:
        cv2.imwrite(os.path.join(OUT_PERSON,img_name),processed)
    else:
        cv2.imwrite(os.path.join(OUT_EMPTY,img_name),processed)

print("Dataset prepared successfully!")
