#coding=utf-8

import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import copy
import scipy.io
import cv2
import matplotlib.pyplot as plt

# from train import config


def plot_sample(x, y, axis):
    """

    :param x: (9216,)
    :param y: (15,2)
    :param axis:
    :return:
    """
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[:,0], y[:,1], marker='x', s=10)

def plot_demo(X,y):
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y[i], ax)

    plt.show()


class MPIIDataset(Dataset):
    def __init__(self,config,state='train'):
        """
        :param X: (N,96*96)
        :param gts: (N,15,2)
        """
        self.__sigma = config['sigma']
        self.__debug_vis = config['debug_vis']
        #self.__fname = config['fname']
        self.__is_test = config['is_test']
        self.image_root = config['image_root']
        self.annos = scipy.io.loadmat(config['fname'])[state][0]
        self.out_width = config['out_width']
        self.in_width = config['in_width']
        self.point_size = config['point_size']
        self.nclass = config['nclass']
        self.pnum = 0

        x_range = [i for i in range(self.point_size)]
        y_range = [i for i in range(self.point_size)]
        xx, yy = np.meshgrid(x_range, y_range)
        d2 = (xx - (self.point_size-1)/2) ** 2 + (yy - (self.point_size-1)/2) ** 2
        exponent = d2 / 2.0 / self.__sigma / self.__sigma
        self.heatmap = np.exp(-exponent)
        #np.savetxt('heatmap.txt',self.heatmap)
        self.heatmap = np.pad(self.heatmap,self.out_width,'constant')
        heatmap_show = self.heatmap.astype(np.uint8)*255
        

    def __len__(self):
        return len(self.annos)

    def putGaussian(self,gt,x,y):
        xx = self.out_width+(self.point_size-1)//2-y
        yy = self.out_width+(self.point_size-1)//2-x
        #print(xx,yy)
        gt=gt+self.heatmap[xx:xx+self.out_width, yy:yy+self.out_width]
        #gt_show = gt.astype(np.uint8)*255
        #cv2.imshow('gt',gt_show)
        #cv2.waitKey(0)
        #self.pnum += 1
        return gt

    def __getitem__(self, item):
        #print(item)
        image_name = self.annos['image'][item][0]
        img = cv2.imread(self.image_root+image_name)
        w,h = img.shape[0],img.shape[1]
        #print(w,h)
        img = cv2.resize(img,(self.in_width,self.in_width))
        #print('debug#1')
        img = np.transpose(img,(2,0,1)).astype(np.float32)
        #print('debug#2')
        persons = self.annos['annorect'][item][0]
        #print('debug#4')
        gt = np.zeros((self.nclass,self.out_width,self.out_width))
        self.pnum = 0
        #print(gt.shape)
        for person in persons:
            #if(person['annopoints'].shape[0]==0):
            #    print(image_name)
            person = person['annopoints'][0]['point'][0][0]
            #print(len(person))
            for point in person:
                pid = point['id'][0][0]
                x = int(point['x'][0][0]*self.out_width/h)
                y = int(point['y'][0][0]*self.out_width/w)
                if(x<0 or y<0 or x>self.out_width-1 or y>self.out_width-1):
                    continue
                #print('true x,y is:{},{}\n'.format(x,y))
                gt[pid] = self.putGaussian(gt[pid],x,y)
                #np.savetxt(str(pid)+'.txt',gt[pid][y-9:y+9,x-9:x+9])
        #print(gt.shape)
        gt=gt.astype(np.float32)
        #print(self.pnum)
        return img,gt

    def visualize_heatmap_target(self,oriImg,heatmap,stride):

        plt.imshow(oriImg)
        plt.imshow(heatmap, alpha=.5)
        plt.show()

if __name__ == '__main__':
    from train import config
    dataset = MPIIDataset(config)
    dataLoader = DataLoader(dataset=dataset,batch_size=1,shuffle=False)
    for i, (img,gts) in enumerate(dataLoader):
        img,gts = img.numpy()[0],gts.numpy()[0]
        gt=np.zeros((1,256,256)).astype(np.float32)
        for g in gts:
            gt=gt+[g]
            #peakIndexTuple = np.unravel_index(np.argmax(g), g.shape)
            
        #np.savetxt('gt1.txt',gt[0][164:176,150:162])
        #np.savetxt('gt2.txt',gt[0][91:103,163:175])
        #np.savetxt('gt3.txt',gt[0][169:181,164:176])
        #img,gt = np.transpose(img,(1,2,0)).astype(np.uint8),np.transpose(gt,(1,2,0)).astype(np.uint8)*255
        img,gt = np.transpose(img,(1,2,0)).astype(np.uint8), np.transpose(gt,(1,2,0))
        #print(gt.max())
        cv2.imshow('img',img)
        cv2.imshow('gt',gt)
        cv2.waitKey(0)


