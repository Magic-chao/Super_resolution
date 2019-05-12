import os
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data

def RGB_np2Tensor(imgIn, imgTar):
    ts = (2, 0, 1)
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)
    imgTar = torch.Tensor(imgTar.transpose(ts).astype(float)).mul_(1.0)  
    return imgIn, imgTar

def augment(imgIn, imgTar):
    if random.random() < 0.3:
        imgIn = imgIn[:, ::-1, :]
        imgTar = imgTar[:, ::-1, :]
    if random.random() < 0.3:
        imgIn = imgIn[::-1, :, :]
        imgTar = imgTar[::-1, :, :]
    return imgIn, imgTar

def getPatch(imgIn, imgTar, args, scale):
    (ih, iw, c) = imgIn.shape
    (th, tw) = (scale * ih, scale * iw)
    tp = args.patchSize
    ip = tp // scale
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    imgIn = imgIn[iy:iy + ip, ix:ix + ip, :]
    imgTar = imgTar[ty:ty + tp, tx:tx + tp, :]
    return imgIn, imgTar

class DIV2K(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.scale = args.scale
        apath = args.dataDir
        self.dirHR = args.HRDir
        self.dirHR_refer = 'refer_128x128'
        self.dirLR = '32x32'
        self.dirHR = os.path.join(apath, self.dirHR)
        # self.dirTar = os.path.join(apath, dirHR)
        self.fileList = os.listdir( self.dirHR )
        self.nTrain = len(self.fileList)
        
    def __getitem__(self, idx):
        scale = self.scale
        nameHr, nameLR, dirHR_refer = self.getFileName(idx)
        imgHr = cv2.imread(nameHr)
        imgLR = cv2.imread(nameLR)
        imgHR_refer = cv2.imread(dirHR_refer)
        # if self.args.need_patch:
        # if self.args.need_patch:
        #     imgIn, imgTar = getPatch(imgIn, imgTar, self.args, 2)
        # # imgIn, imgTar = augment(imgIn, imgTar)
        # # return RGB_np2Tensor(imgIn, imgTar)
        imgHr = imgHr / 256.0 - 0.5
        imgHr = imgHr.transpose((2, 0, 1))

        imgLR = imgLR / 256.0 - 0.5
        imgLR = imgLR.transpose((2, 0, 1))

        imgHR_refer = imgHR_refer / 256.0 - 0.5
        imgHR_refer = imgHR_refer.transpose((2, 0, 1))
        return imgHr, imgLR, imgHR_refer

    def __len__(self):
        return self.nTrain   
        
    def getFileName(self, idx):
        name = self.fileList[idx]
        nameHr = os.path.join(self.dirHR, name)
        nameLR = os.path.join(self.dirLR, name)
        dirHR_refer = os.path.join(self.dirHR_refer, name)

        return nameHr, nameLR,dirHR_refer
