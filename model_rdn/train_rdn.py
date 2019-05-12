import argparse
import math
import cv2
import numpy as np
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
parser.add_argument('--model_name', default='RDN_0', help='model directory')
parser.add_argument('--finetuning', default=True, help='finetuning the training')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')

# ###################################################################### 数据路径
parser.add_argument('--result_SR_Dir', default='E:/ali_uku/round1_train_result', help='datasave directory')
parser.add_argument('--LR_Dir', default='round1_train_input', help=' directory')
parser.add_argument('--HR_Dir', default='round1_train_label', help=' directory')
parser.add_argument('--dataDir', default='E:/ali_uku', help='dataset directory')

parser.add_argument('--need_patch', default=True, help='get patch form image')
parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=256, help='patch size')
parser.add_argument('--nThreads', type=int, default=8, help='number of threads for data loading')  # 线程数

parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--lrDecay', type=int, default=500, help='input LR video')
parser.add_argument('--decayType', default='step', help='output SR video')
parser.add_argument('--lossType', default='L1', help='output SR video')
parser.add_argument('--scale', type=int, default=4, help='scale output size /input size')
args = parser.parse_args()

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data)

from data.datax4 import vedio_data  # x8

def get_dataset(args):
    data_train = vedio_data(args)
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batchSize,
                                             drop_last=True, shuffle=True, num_workers=int(args.nThreads),
                                             pin_memory=False)
    return dataloader

def set_loss(args):
    lossType = args.lossType
    if lossType == 'MSE':
        lossfunction = nn.MSELoss()
    # if lossType == 'L2':
    # 	lossfunction = nn.L
    elif lossType == 'L1':
        lossfunction = nn.L1Loss()
    return lossfunction

def set_lr(args, epoch, optimizer):
    lrDecay = args.lrDecay
    decayType = args.decayType
    if decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2 ** epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'inv':
        k = 1 / lrDecay
        lr = args.lr / (1 + k * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(args):
    #  select network
    # if args.model_name == 'RDN':    # 模型
    my_model = RDN(args)  # model.RDN()
    my_model.apply(weights_init_kaiming)

    save = saveData(args)
    # fine-tuning or retrain
    if args.finetuning:
        model_path =os.path.join(args.model_savepath,args.model_name)
        my_model = save.load_model(my_model, model_path)  # hh
    # load data
    my_model.cuda()
    my_model.train()
    dataloader = get_dataset(args)
    L1_lossfunction = set_loss(args)
    total_loss = 0

    writer = SummaryWriter(log_dir="./logs/", comment='loss')
    for epoch in range(args.epochs):            #开始训练
        optimizer = optim.Adam(my_model.parameters())
        learning_rate = set_lr(args, epoch, optimizer)
        total_loss_ = 0
        for i, (lr_in, hr_in, name) in enumerate(dataloader):
            im_lr = Variable(lr_in.cuda().float(), volatile=False)
            im_hr = Variable(hr_in.cuda().float(), volatile=False)
            # im_refer = Variable(refer.cuda().float(), volatile=False)
            my_model.zero_grad()
            out_put = my_model(im_lr)  # my_model(im_lr, im_refer)
            L1_loss = L1_lossfunction(out_put, im_hr)
            # if i % 1000 == 0:
            #     print("############################epoch{}___itera__{}".format(epoch, i))
            #     lr = lr_in[1].squeeze()
            #     lr = np.array(lr).transpose((1, 2, 0))
            #     lr = lr + 0.5
            #     lr = np.ceil(lr * 256)
            #     cv2.imwrite('./out/train/epoch{}_{}realsmall.jpg'.format(epoch, i), lr)
            #
            #     hr = hr_in[1].squeeze()
            #     hr = np.array(hr).transpose((1, 2, 0))
            #     hr = hr + 0.5
            #     hr = np.ceil(hr * 256)
            #     cv2.imwrite('./out/train/epoch{}_{}real.jpg'.format(epoch, i), hr)
            #
            #     refer = refer[1].squeeze()
            #     refer = np.array(refer).transpose((1, 2, 0))
            #     refer = refer + 0.5
            #     refer = np.ceil(refer * 256)
            #     cv2.imwrite('./out/train/epoch{}_{}refer.jpg'.format(epoch, i), refer)
            #
            #     img_hr_out = out_put[1].squeeze()
            #     img_hr_out = img_hr_out.cpu().data.numpy()
            #     img_hr_out = img_hr_out.transpose((1, 2, 0))
            #     img_hr_out = img_hr_out + 0.5
            #     img_hr_out = np.ceil(img_hr_out * 256)
            #     cv2.imwrite('./out/train/epoch{}_{}.jpg'.format(epoch, i), img_hr_out)
            L1_loss.backward()
            writer.add_scalar('mse_loss', L1_loss, epoch * len(dataloader) + i)
            optimizer.step()
            total_loss += L1_loss.data.cpu().numpy()
            if i%100==0:
                print("##"*20,i)
            print(i)
        total_loss = total_loss / (i + 1)

        print("########################################################{}".format(i))
        if (epoch + 1) % 1 == 0:
            log = "[{} / {}] \tLearning_rate: {}\t total_loss: {:.4f}\t".format(epoch + 1,
                                                                                args.epochs, learning_rate, total_loss_)
            print(log)
            # save.save_log(log)
            save.save_model(my_model, epoch, args.model_savepath)

    writer.close()

if __name__ == '__main__':
    train(args)

    # torch.save(
    # 	        model.state_dict(),
    # 	        self.save_dir_model + '/model_{}_{}.pt'.format(4_10,epoch)
