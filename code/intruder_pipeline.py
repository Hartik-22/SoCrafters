import cv2
import numpy as np

def preprocess(frame):

    frame = cv2.resize(frame,(64,64))

    blur = cv2.GaussianBlur(frame,(5,5),0)

    sobelx = cv2.Sobel(blur,cv2.CV_32F,1,0,ksize=3)
    sobely = cv2.Sobel(blur,cv2.CV_32F,0,1,ksize=3)

    mag = cv2.magnitude(sobelx,sobely)

    # normalize instead of raw edges
    mag = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    mag = np.uint8(mag)

    return mag
