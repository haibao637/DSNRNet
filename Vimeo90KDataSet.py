'''
Vimeo90K dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

from  utils import *
try:
    import mc  # import memcached
except ImportError:
    pass
logger = logging.getLogger('base')


class Vimeo90KDataset(data.Dataset):
    '''
    Reading the training Vimeo90K dataset
    key example: 00001_0001 (_1, ..., _7)
    GT (Ground-Truth): 4th frame;
    LQ (Low-Quality): support reading N LQ frames, N = 1, 3, 5, 7 centered with 4th frame
    '''

    def __init__(self,dataroot_GT,dataroot_LQ,mode='train'):
        super(Vimeo90KDataset, self).__init__()

        # temporal augmentation

        # logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
        #     ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.mode = mode
        self.dataroot_GT = dataroot_GT
        self.dataroot_LQ = dataroot_LQ
        #### determine the LQ frame list
        '''
        N | frames
        1 | 4
        3 | 3,4,5
        5 | 2,3,4,5,6
        7 | 1,2,3,4,5,6,7
        '''
        self.LQ_frames_list = []
        for i in range(7):
            self.LQ_frames_list.append(i + (9 - 7) // 2)
        self.paths_GT, _ = get_image_paths( self.dataroot_GT)
        logger.info('Using lmdb meta info for cache keys.')


    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.dataroot_GT, readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.dataroot_LQ, readonly=True, lock=False, readahead=False,
                                meminit=False)

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def _read_img_mc(self, path):
        ''' Return BGR, HWC, [0, 255], uint8'''
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def __getitem__(self, index):
        self._init_lmdb()

        scale = 4
        GT_size = 64*4
        key = self.paths_GT[index]
        name_a, name_b = key.split('_')
        img_GT = read_img(self.GT_env, key + '_4', (3, 256, 448))
        #### get LQ images
        LQ_size_tuple = (3, 64, 112)# if self.LR_input else (3, 256, 448)
        img_LQ_l = []
        for v in self.LQ_frames_list:
            img_LQ = read_img(self.LQ_env, key + '_{}'.format(v), LQ_size_tuple)

            img_LQ_l.append(img_LQ)

        if self.mode == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop

            LQ_size = GT_size // scale
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
            # augmentation - flip, rotate
            img_LQ_l.append(img_GT)
            rlt = augment(img_LQ_l)
            img_LQ_l = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        return img_LQs,img_GT#{'LQs': img_LQs, 'GT': img_GT}

    def __len__(self):
        return len(self.paths_GT)

if __name__ == "__main__":
    dataset = Vimeo90KDataset("/home/haibao637/xdata/vimeo90k/vimeo90k_train_GT.lmdb","/home/haibao637/xdata/vimeo90k/vimeo90k_train_LR7frames.lmdb")
    # print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    for lr,hr in dataloader:
        print(lr.shape,hr.shape)
        break
