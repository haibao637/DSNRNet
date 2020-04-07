import torch
from PIL import  Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import  DataLoader
import os
import torchvision.transforms as trans
class SEDataSet(Dataset):
    def __init__(self,video_dir):
        videos = os.listdir(video_dir)
        videos = [os.path.join(video_dir,video) for video in videos]
        videos = [video for video in videos if len(os.listdir(video))>100]
        # videos = [video ]
        self.sequences =[]
        for video in videos:
            self.sequences.extend([os.path.join(video,img_name) for img_name in os.listdir(video) if img_name.find(".png")])
        self.transform=trans.Compose([
            # trans.Resize((448,448)),
            # trans.RandomCrop((896,896)),
            trans.RandomCrop((224,224)),
            trans.Grayscale(),
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

if __name__=="__main__":
    dataset = SEDataSet("/home/haibao637/Downloads/")
    dataloader = DataLoader(dataset,batch_size=1)
    for img in dataloader:
        print(img.min(),img.max())