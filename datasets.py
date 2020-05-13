import torch
from PIL import  Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import  DataLoader
import os
import torchvision.transforms as trans
import cv2
import random
import numpy as np
import torch.nn.functional as F
class SEDataSet(Dataset):
    def __init__(self,video_dir):
        videos = os.listdir(video_dir)
        videos = [os.path.join(video_dir,video) for video in videos]
        videos = [video for video in videos if len(os.listdir(video))>100]
        # videos = [video ]
        self.sequences =[]
        for video in videos:
            seq = [os.path.join(video,img_name) for img_name in os.listdir(video) if img_name.find(".png")]
            shape = cv2.imread(seq[0]).shape
            if (shape[0]>=224) and (shape[1]>=224):
                self.sequences.extend(seq)
        self.transform=trans.Compose([
            # trans.Resize((448,448)),
            trans.RandomCrop((224,224)),
            # trans.RandomCrop((224,224)),
            # trans.Grayscale(),
            trans.ToTensor()

        ])
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, item):
        img = Image.open(self.sequences[item])
        tensor = self.transform(img)
        # mean = torch.mean(tensor,dim=[1,2],keepdim=True)
        # var = torch.std(tensor,dim=[1,2],keepdim=True)
        return tensor
def random_crop(lr, hr, lr_size, scale):
    """
    Random crop a patch from hr and lr with the same location
    Image should be numpy array, dimension NCHW (N, C is optional)
    """
    h, w = lr.shape[-2:]
    assert h >= lr_size and w >= lr_size
    assert lr.shape[-2] * scale == hr.shape[-2] # height match
    assert lr.shape[-1] * scale == hr.shape[-1] # width match
    x = random.randint(0, w - lr_size)
    y = random.randint(0, h - lr_size)

    hr_size = lr_size * scale
    hr_x, hr_y = x*scale, y*scale

    crop_lr = lr[..., y:y+lr_size, x:x+lr_size]
    crop_hr = hr[..., hr_y:hr_y+hr_size, hr_x:hr_x+hr_size]

    return crop_lr, crop_hr
class SRDataSet(Dataset):
    def __init__(self,hr_dir,lr_dir,mode='train'):
        # hr_dir = os.path.join(video_dir,"lr","HR")
        # lr2_dir = os.path.join(video_dir,"lr","LRs4")
        # lr2_1_dir = os.path.join(video_dir,"lr","LRs2")

        self.mode = mode
        videos = os.listdir(hr_dir)
        hr_videos = [os.path.join(hr_dir,video) for video in videos]
        lr_videos = [os.path.join(lr_dir,video) for video in videos]
        # videos = os.listdir(video_dir)
        # videos = [os.path.join(video_dir,video) for video in videos]
        # videos = [video for video in videos if len(os.listdir(video))>100]
        # videos = [video ]

        self.hr_seqs =[]
        self.lr_seqs = []
        for lr_video,hr_video in zip(lr_videos,hr_videos):

            lr_seq = sorted([img_name for img_name in os.listdir(lr_video) if img_name.find(".png")])
            hr_seq = [os.path.join(hr_video,img_name) for img_name in  lr_seq]
            lr_seq = [os.path.join(lr_video,img_name) for img_name in lr_seq]

            # lr_seq.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))
            # hr_seq.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))

            # shape = cv2.imread(seq[0]).shape
            lr_seqs =  [lr_seq[i:i+5] for i in range(len(lr_seq)-5)]
            hr_seqs = [hr_seq[i:i+5] for i in range(len(hr_seq)-5)]
            # if mode == 'train':

            #     lr_seqs = lr_seqs + [[lr_seq[i]]*5 for i in range(len(lr_seq))]
            #     lr_seqs = lr_seqs + [lr_seq[i:i+10:2] for i in range(len(lr_seq)-10)]

            #     hr_seqs = hr_seqs + [[hr_seq[i]]*5 for i in range(len(hr_seq))]
            #     hr_seqs = hr_seqs + [hr_seq[i:i+10:2] for i in range(len(hr_seq)-10)]

            self.hr_seqs.extend(hr_seqs)
            self.lr_seqs.extend(lr_seqs)
            # if (shape[0]>=448) and (shape[1]>=448):
            #     seq = list(zip(seq[:-4],seq[1:-3],seq[2:-2],seq[3:-1],seq[4:]))
            #     self.sequences.extend(seq)
        self.hr_trans=trans.Compose([
            # trans.Resize((448,448)),
            # trans.CenterCrop((180,320)),
            # trans.CenterCrop((256,256)),
            # trans.Grayscale(),
            trans.ToTensor()

        ])

        self.lr_trans=trans.Compose([
            # trans.Resize((448,448)),
            # trans.CenterCrop((64,64)),
            # trans.RandomCrop((224,224)),
            # trans.Grayscale(),
            trans.ToTensor()

        ])


        self.val_trans = trans.Compose([
            # trans.Resize((448,448)),
            # trans.CenterCrop((180,320)),
            # trans.CenterCrop((256,256)),
            # trans.Grayscale(),
            trans.ToTensor()

        ])

        # self.resize =
    def __len__(self):
        return len(self.hr_seqs)
    def __getitem__(self, item):
        # print(self.hr_seqs[item],self.lr_seqs[item])
        # if self.mode =='train':
        #     trans = self.hr_trans
        # else:
        #     trans = self.cent_trans
        if self.mode =='val':
            hr_trans = self.val_trans
            lr_trans = self.val_trans
        else:
            hr_trans = self.hr_trans
            lr_trans = self.lr_trans
        # hr = [hr_trans(Image.open(img_path)) for img_path in self.hr_seqs[item]]
        # hr = torch.stack(hr,0)#c,v,h,w
        # mode = ["bilinear","bicubic"][random.randint(0,1)]
        # mode = "bicubic"
        # lr = F.interpolate(hr,scale_factor=0.25,align_corners=False,mode="bicubic").clamp(0,1.0)

        lr = [lr_trans(Image.open(img_path)) for img_path in self.lr_seqs[item]]
        lr = torch.stack(lr,0)#v,c,h,w
        hr = hr_trans(Image.open(self.hr_seqs[item][lr.shape[0]//2]))

        if self.mode == 'train':
            lr,hr = random_crop(lr,hr,lr_size=64,scale = 4)
        # print(hr.shape,lr.shape)
        # mean = torch.mean(tensor,dim=[1,2],keepdim=True)
        # var = torch.std(tensor,dim=[1,2],keepdim=True)
        return lr,hr


if __name__=="__main__":
    dataset = SEDataSet("/home/haibao637/Downloads/")
    dataloader = DataLoader(dataset,batch_size=1)
    for img in dataloader:
        print(img.min(),img.max())