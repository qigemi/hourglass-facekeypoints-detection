#coding=utf-8
import argparse
import torch
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
import pprint

from data_loader import MPIIDataset
from models import HGNet

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='SSD_vgg',
                    help='RFB_vgg ,RFB_E_vgg RFB_mobile SSD_vgg version.')
parser.add_argument('-s', '--size', default='512',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='COCO',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=8,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--resume_net', default=False, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')

parser.add_argument('-max', '--max_epoch', default=300,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('-we', '--warm_epoch', default=1,
                    type=int, help='max epoch for retraining')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='weights/',
                    help='Location to save checkpoint models')
parser.add_argument('--date', default='1213')
parser.add_argument('--save_frequency', default=10)
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--test_frequency', default=10)

args = parser.parse_args()

config = dict()
config['lr'] = 0.1
config['momentum'] = 0.9
config['weight_decay'] = 1e-4
config['epoch_num'] = 10
config['batch_size'] = 1
config['sigma'] = 1.
config['debug_vis'] = False         # 是否可视化heatmaps
config['fname'] = 'data/annolist/train_gt.mat'
config['image_root'] = 'data/images/'
config['in_width'] = 128
config['out_width'] = 128
config['nclass'] = 16
config['point_size'] = 5 #奇数
# config['fname'] = 'data/training.csv'
# config['is_test'] = False
config['is_test'] = True
config['save_freq'] = 10
config['checkout'] = ''
config['start_epoch'] = 0
config['eval_freq'] = 5
config['debug'] = False
config['lookup'] = 'data/IdLookupTable.csv'
config['featurename2id'] = {
    'left_eye_center_x':0,
    'left_eye_center_y':1,
    'right_eye_center_x':2,
    'right_eye_center_y':3,
    'left_eye_inner_corner_x':4,
    'left_eye_inner_corner_y':5,
    'left_eye_outer_corner_x':6,
    'left_eye_outer_corner_y':7,
    'right_eye_inner_corner_x':8,
    'right_eye_inner_corner_y':9,
    'right_eye_outer_corner_x':10,
    'right_eye_outer_corner_y':11,
    'left_eyebrow_inner_end_x':12,
    'left_eyebrow_inner_end_y':13,
    'left_eyebrow_outer_end_x':14,
    'left_eyebrow_outer_end_y':15,
    'right_eyebrow_inner_end_x':16,
    'right_eyebrow_inner_end_y':17,
    'right_eyebrow_outer_end_x':18,
    'right_eyebrow_outer_end_y':19,
    'nose_tip_x':20,
    'nose_tip_y':21,
    'mouth_left_corner_x':22,
    'mouth_left_corner_y':23,
    'mouth_right_corner_x':24,
    'mouth_right_corner_y':25,
    'mouth_center_top_lip_x':26,
    'mouth_center_top_lip_y':27,
    'mouth_center_bottom_lip_x':28,
    'mouth_center_bottom_lip_y':29,
}

def get_peak_points(heatmaps):
    """
    :param heatmaps: numpy array (N,15,96,96)
    :return:numpy array (N,15,2)
    """
    N,C,H,W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i,j] == heatmaps[i,j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x,y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points

def get_mse(pred_points,gts,indices_valid=None):
    """

    :param pred_points: numpy (N,15,2)
    :param gts: numpy (N,15,2)
    :return:
    """
    pred_points = pred_points[indices_valid[0],indices_valid[1],:]
    gts = gts[indices_valid[0],indices_valid[1],:]
    pred_points = Variable(torch.from_numpy(pred_points).float(),requires_grad=False)
    gts = Variable(torch.from_numpy(gts).float(),requires_grad=False)
    criterion = nn.MSELoss()
    loss = criterion(pred_points,gts)
    return loss

def calculate_mask(heatmaps_target):
    """

    :param heatmaps_target: Variable (N,15,96,96)
    :return: Variable (N,15,96,96)
    """
    N,C,_,_ = heatmaps_targets.size()
    N_idx = []
    C_idx = []
    for n in range(N):
        for c in range(C):
            max_v = heatmaps_targets[n,c,:,:].max().data[0]
            if max_v != 0.0:
                N_idx.append(n)
                C_idx.append(c)
    mask = Variable(torch.zeros(heatmaps_targets.size()))
    mask[N_idx,C_idx,:,:] = 1.
    mask = mask.float().cuda()
    return mask,[N_idx,C_idx]

if __name__ == '__main__':
    pprint.pprint(config)
    torch.manual_seed(0)
    cudnn.benchmark = True
    net = HGNet()

    def init_weights(m):
        print(m)
        if type(m) == nn.Linear:
            m.weight.data.fill_(1.0)
            print(m.weight)
    net.apply(init_weights)
    #print(net.parameters())
    net.float().cuda()
    net.train()
    criterion = nn.MSELoss(size_average=False).cuda()

    optimizer = optim.SGD(net.parameters(), lr=config['lr'],momentum=config['momentum'], weight_decay=config['weight_decay'])
    trainDataset = MPIIDataset(config)
    
    trainDataLoader = DataLoader(trainDataset,config['batch_size'],True)
    sample_num = len(trainDataset)

    if (config['checkout'] != ''):
        net.load_state_dict(torch.load(config['checkout']))

    for epoch in range(config['start_epoch'],config['epoch_num']+config['start_epoch']):
        running_loss = 0.0
        for i, (inputs, gts) in enumerate(trainDataLoader):
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()
            
            optimizer.zero_grad()
            print(inputs.shape)
            outputs = net(inputs)
            
            print(outputs.shape,gts.shape)
            loss = criterion(outputs, gts)
            loss.backward()
            optimizer.step()

            # 统计最大值与最小值
            v_max = torch.max(outputs)
            v_min = torch.min(outputs)

            outputs = outputs.cpu().detach().numpy()
            gts = gts.cpu().detach().numpy()
            result = np.zeros((config['out_width'],config['out_width']))
            groundtruth = np.zeros((config['out_width'],config['out_width']))
            for i in range(len(outputs[0])):
                result = result+outputs[0][i]
                groundtruth += gts[0][i]
            cv2.imshow('result',result)
            cv2.imshow('groundtruth',groundtruth)
            #cv2.imshow('channel0',outputs[0][0])
            #cv2.imshow('channel1',outputs[0][1])
            #cv2.imshow('channel2',outputs[0][2])
            #cv2.imshow('channel0-1',outputs[0][0]-outputs[0][1])
            cv2.waitKey(500)

            print('[ Epoch {:005d} -> {:005d} / {} ] loss : {:10} max : {:5} min : {}'.format(
                epoch, i * config['batch_size'],
                sample_num, loss.data[0],v_max.data[0],v_min.data[0]))


        if (epoch+1) % config['save_freq'] == 0 or epoch == config['epoch_num'] - 1:
            torch.save(net.state_dict(),'kd_epoch_{}_model.ckpt'.format(epoch))

