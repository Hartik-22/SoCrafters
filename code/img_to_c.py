import cv2
import numpy as np
img = cv2.imread(r"C:\Users\Hartik Rai\OneDrive\Desktop\ARM_Project\code\test.jpg",0)

if img is None:
    print("Image not found")
    exit()

img = cv2.resize(img,(64,64))

flat = img.flatten()

with open("test_img.h","w") as f:
    f.write("unsigned char input_img[4096] = {\n")
    for i,val in enumerate(flat):
        f.write(str(int(val)))
        if i!=len(flat)-1:
            f.write(",")
        if i%32==0:
            f.write("\n")
    f.write("\n};")

print("Header file created")
