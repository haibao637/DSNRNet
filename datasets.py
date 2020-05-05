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

class SRDataSet(Dataset):
    def __init__(self,video_dir):
        # hr_dir = os.path.join(video_dir,"lr","HR")
        # lr2_dir = os.path.join(video_dir,"lr","LRs4")
        # lr2_1_dir = os.path.join(video_dir,"lr","LRs2")
        hr_dir = video_dir
        videos = os.listdir(hr_dir)
        hr_videos = [os.path.join(hr_dir,video) for video in videos]
        # lr_videos = [os.path.join(lr2_dir,video) for video in videos]
        # videos = os.listdir(video_dir)
        # videos = [os.path.join(video_dir,video) for video in videos]
        # videos = [video for video in videos if len(os.listdir(video))>100]
        # videos = [video ]

        self.hr_seqs =[]
        # self.lr_seqs = []
        for hr_video in hr_videos:
            # lr_seq = sorted([os.path.join(lr_video,img_name) for img_name in os.listdir(lr_video) if img_name.find(".png")])

            hr_seq = sorted([os.path.join(hr_video,img_name) for img_name in os.listdir(hr_video) if img_name.find(".png")])

            # shape = cv2.imread(seq[0]).shape
            # lr_seqs = [lr_seq[i:i+3] for i in range(len(lr_seq)-3)]
            hr_seqs = [hr_seq[i:i+3] for i in range(len(hr_seq)-3)]
            self.hr_seqs.extend(hr_seqs)
            # self.lr_seqs.extend(lr_seqs)
            # if (shape[0]>=448) and (shape[1]>=448):
            #     seq = list(zip(seq[:-4],seq[1:-3],seq[2:-2],seq[3:-1],seq[4:]))
            #     self.sequences.extend(seq)
        self.hr_transform=trans.Compose([
            # trans.Resize((448,448)),
            # trans.CenterCrop((180,320)),
            trans.CenterCrop((256,256)),
            # trans.Grayscale(),
            trans.ToTensor()

        ])

        self.lr_transform=trans.Compose([
            # trans.Resize((448,448)),
            trans.CenterCrop((64,64)),
            # trans.RandomCrop((224,224)),
            # trans.Grayscale(),
            trans.ToTensor()

        ])
        self.resize =
    def __len__(self):
        return len(self.hr_seqs)
    def __getitem__(self, item):
        # print(self.hr_seqs[item],self.lr_seqs[item])
        hr = [self.hr_transform(Image.open(img_path)) for img_path in self.hr_seqs[item]]
        hr = torch.stack(hr,0)#c,v,h,w
        # mode = ["bilinear","bicubic"][random.randint(0,1)]
        # mode = "bicubic"
        lr = F.interpolate(hr,scale_factor=0.25,align_corners=False,mode="bicubic").clamp(0,1.0)

        # lr = [self.lr_transform(Image.open(img_path)) for img_path in self.lr_seqs[item]]
        # lr = torch.stack(lr,0)#v,c,h,w

        # print(hr.shape,lr.shape)
        # mean = torch.mean(tensor,dim=[1,2],keepdim=True)
        # var = torch.std(tensor,dim=[1,2],keepdim=True)
        return lr,hr


if __name__=="__main__":
    dataset = SEDataSet("/home/haibao637/Downloads/")
    dataloader = DataLoader(dataset,batch_size=1)
    for img in dataloader:
        print(img.min(),img.max())