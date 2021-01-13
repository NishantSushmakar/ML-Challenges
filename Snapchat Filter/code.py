# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 20:28:02 2021

@author: nishant
"""
import cv2
import numpy as np
import pandas as pd

img = cv2.imread('C:/Users/nishant/Snapchat Filter/Test/Before.png')

glasses_img = cv2.imread('C:/Users/nishant/Snapchat Filter/Train/glasses.png',-1)
mustache_img = cv2.imread('C:/Users/nishant/Snapchat Filter/Train/mustache.png',-1) 


eye_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('Nose18x15.xml')


eyes = eye_cascade.detectMultiScale(img,1.4,5)
nose = nose_cascade.detectMultiScale(img,1.3,5)

# For eyes

for (ex, ey, ew, eh) in eyes:
    
    
    ey += 0
    ex -= 20
    glasses_img = cv2.resize(glasses_img,(ew+25,eh+20) )    
    y1, y2 = ey, ey + glasses_img.shape[0]
    x1, x2 = ex, ex + glasses_img.shape[1]
    alpha_s = glasses_img[:, :, 3]/ 255.0
    alpha_l = 1.0 - alpha_s
    
    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha_s * glasses_img[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])
    
# For Mustache

for (nx,ny,nw,nh)  in  nose:
    #cv2.rectangle(img,(nx,ny),(nx+nw,ny+nh),(0,255,255),2)
    ny += 28
    nx += 5
    mustache_img = cv2.resize(mustache_img,(nw,nh) )
    y1, y2 = ny, ny + mustache_img.shape[0]
    x1, x2 = nx, nx + mustache_img.shape[1]
    alpha_s = mustache_img[:, :, 3]/ 255.0
    alpha_l = 1.0 - alpha_s
    
    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha_s * mustache_img[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])
    
    

#cv2.imshow("Image",img)
ans = np.reshape(img,(-1,3))
submission = pd.DataFrame(ans,columns=['Channel 1','Channel 2','Channel 3'])
submission.to_csv('submission.csv',index=False)
#cv2.waitKey(0) cv2.destroyAllWindows()
