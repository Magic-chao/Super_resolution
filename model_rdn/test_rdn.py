import argparse
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
from tensorboardX import SummaryWriter
from model_rdn.RDN import RDN
from utils import *

parser = argparse.ArgumentParser(description='Semantic aware super-resolution')
# ########################################################## 模型加载
parser.add_argument('--model_savepath', default='E:\lunwen\game_alibaba\model_rdn\weight', help='dataset directory')
parser.add_argument('--model_name', default='RDN_9', help='model directory')
parser.add_argument('--finetuning', default=True, help='finetuning the training')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')

# ###################################################################### 数据路径
parser.add_argument('--result_SR_Dir', default='E:/ali_uku/round1_train_result', help='datasave directory')
parser.add_argument('--LR_Dir', default='validation', help=' directory')
parser.add_argument('--HR_Dir', default='round1_train_label', help=' directory')
parser.add_argument('--dataDir', default='E:/ali_uku', help='dataset directory')

parser.add_argument('--need_patch', default=True, help='get patch form image')
parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=256, help='patch size')
parser.add_argument('--nThreads', type=int, default=4, help='number of threads for data loading')  # 线程数

parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--lrDecay', type=int, default=500, help='input LR video')
parser.add_argument('--decayType', default='step', help='output SR video')
parser.add_argument('--lossType', default='L1', help='output SR video')
parser.add_argument('--scale', type=int, default=4, help='scale output size /input size')
args = parser.parse_args()
from tools.psnr import get_psnr
from data.my_data import vedio_data
def get_dataset(args):
    data_train = vedio_data(args)
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batchSize,shuffle=False)
    return dataloader


def test(args):
    #  select network
    # if args.model_name == 'RDN':  # 模型
    my_model = RDN(args)  # model.RDN()

    save = saveData(args)
    dataloader = get_dataset(args)
    my_model.cuda()
    my_model.train()
    model_path = os.path.join(args.model_savepath, args.model_name)
    # my_model.load_state_dict(torch.load(model_path))
    my_model = save.load_model(my_model, model_path)
    for i, (lr_in, name) in enumerate(dataloader):
        im_lr = Variable(lr_in.cuda().float(), volatile=False)
        out_put = my_model(im_lr)

        lr = lr_in[0]
        lr = np.array(lr).transpose((1, 2, 0))
        lr = lr + 0.5
        lr = np.ceil(lr * 256)
        cv2.imwrite(args.result_SR_Dir+"/lr"+'/%05d_small.bmp'%(i), lr)  # test_dataset/predict_x8

        img_hr_out = out_put[0]
        img_hr_out = img_hr_out.cpu().data.numpy()
        img_hr_out = img_hr_out.transpose((1, 2, 0))
        img_hr_out = img_hr_out + 0.5
        img_hr_out = np.ceil(img_hr_out * 256)
        cv2.imwrite(args.result_SR_Dir+"/sr"+'/%05d_sr.bmp'%(i), img_hr_out)

if __name__ == '__main__':
    test(args)
