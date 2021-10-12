# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:47:38 2019

@author: Lenovo
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, filters

def main():
    print(cv2.__version__)
    
if __name__=='__main__':
    main()
    


def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print('event',x,y)
        print(cv2.cvtColor(np.uint8([[img2[y,x]]]),cv2.COLOR_BGR2HSV))
        #cv2.circle(img,(x,y),80,(255,255,0),1)

fpath = 'c:/users/lenovo/desktop/data'

def read(fpath):
    imgs = []
    
    for path,subdir,files in os.walk(fpath):
        extl = ['jpg','jpeg','PNG']
        for file in files:
            for ext in extl:
                if file.find(ext)>0:
                    imgs.append(file)
                    break
        for img in imgs:
            ipath = path + '/' + img
            print(ipath)
            img = cv2.imread(ipath)
            cv2.namedWindow('image', cv2.WINDOW_FREERATIO)
            cv2.imshow('image',img)
            k = cv2.waitKey(1000) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break
        cv2.destroyAllWindows()
        print(imgs)
        break                  #don't go to sub-directories

img = cv2.imread('C:/Users/Lenovo/Desktop/data/kana.jpg')
img2 = cv2.imread('C:/Users/Lenovo/Desktop/data/suzu.jpg')

def mouse_draw(img):
    events = [i for i in dir(cv2) if 'EVENT' in i]
    print(events)
    
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('image',draw_circle)   #calls draw method
    
    while(True):
        cv2.imshow('image',img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def maskimpose(img,img2):
    
    i2c = img2.copy()
    roi = img2[700:1100,60:476,:]
    i1 = img[:400,150:,:]
    gimg = cv2.cvtColor(i1,cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gimg,np.array([0]),np.array([125]))
    mask = cv2.bitwise_not(mask)
    
    roi[mask==0] = 0
    i1[mask!=0] = 0
    
    dst = cv2.add(roi,i1)
    roi[:,:,:] = dst
    
    #detect hands to print only on t-shirt
    icmask = cv2.inRange(cv2.cvtColor(i2c,cv2.COLOR_BGR2HSV),np.array([2,10,200]),np.array([16,200,255]))
    img2[icmask!=0] = 0
    i2c[icmask==0] = 0
    img2 = cv2.add(img2,i2c)
    
    
    cv2.namedWindow('res', cv2.WINDOW_FREERATIO)
    cv2.imshow('res',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite('suzukana.jpg',img2)

def thresholding(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,th1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,95,1)
    lap = cv2.Laplacian(img,-1)
    
    plt.imshow(th1,cmap='gray',interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    cv2.namedWindow('res', cv2.WINDOW_FREERATIO)
    cv2.imshow('res',lap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

thresholding(img)

#img2 = cv2.merge((img[:,:,2],img[:,:,1],img[:,:,0]))   #change bgr to rgb
#nimg = cv2.add(img/2,img2/2)
#nimg = np.uint8(nimg)


