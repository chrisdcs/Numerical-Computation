"""
Author:
    Chris Ding

Date:
    12/21/2020

Description:
    Apply SVD to an image and reconstruct image by singular vectors

"""

import numpy as np
import cv2

image = cv2.imread(r'Numerical-Computation\SVD\dog.jpg')
# test if image is read successfully
"""
while True:
    cv2.imshow('img',image)
    if cv2.waitKey()==ord('q'):
        break
"""

# loop over image color channels
for i in range():
    USVi = np.linalg.svd(image[:,:,i])