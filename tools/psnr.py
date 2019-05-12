#coding=utf-8
'''
Created on 2018年6月21日

@author: luzhongshan
'''
#codingPSNR

'''
Created on 2018年6月14日

@author: luzhongshan
'''
from PIL import Image
import numpy
import math
import matplotlib.pyplot as plt
#导入你要测试的图像
def get_psnr(y,x):     #行  列  通道
    im = y
    im2 = x
    # print (im.shape,im.dtype)
#图像的行数
    height = im.shape[0]
#图像的列数
    width = im.shape[1]

#提取R通道
    r = im[:,:,0]
#提取g通道
    g = im[:,:,1]
#提取b通道
    b = im[:,:,2]
#打印g通道数组
#print (g)
#图像1,2各自分量相减，然后做平方；
    R = im[:,:,0]-im2[:,:,0]
    G = im[:,:,1]-im2[:,:,1]
    B = im[:,:,2]-im2[:,:,2]
#做平方
    mser = R*R
    mseg = G*G
    mseb = B*B
#三个分量差的平方求和
    SUM = mser.sum() + mseg.sum() + mseb.sum()
    MSE = SUM / (height * width * 3)
    MSE=math.fabs(MSE)
    PSNR = 10*math.log((255.0*255.0/((MSE)*1.0)),10)
    return PSNR

import cv2
import os
if __name__=="__main__":

    str1="E:/ali_uku/validation\youku_00150_00199_h_GT_pic\Youku_00150_h_GT"
    str2="E:/ali_uku/validation\youku_00150_00199_h_GT_pic\Youku_00150_h_GT_sr"
    list1=os.listdir(str1)
    list2=os.listdir(str2)
    list=[]
    for i in range(100):
        img1 = cv2.imread(os.path.join(str1,(list1[i])))
        img2 = cv2.imread(os.path.join(str2,(list2[i])))
        list.append(get_psnr(img1,img2))
    print(list)
    print(sum(list)/len(list))