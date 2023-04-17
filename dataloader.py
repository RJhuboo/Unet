import torch
import os
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from torch.utils.data import Dataset, DataLoader

class Datasets(Dataset):
    def __init__(self,image_dir, label_dir, opt):
        self.opt = opt
        self.imagename = os.listdir(image_dir)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.upsample=False
        
    def __len__(self):
        return len(self.image_dir)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = io.imread(os.path.join(self.image_dir,self.imagename[idx])) # Loading Image
        label = io.imread(os.path.join(self.label_dir,self.imagename[idx]))
        if self.upsample == True or 'lr' in self.imgname:
            image = transform.rescale(image,2)
            image = (image>0.5)*255
            mask_name = os.path.join(self.mask_dir,str(self.labels.iloc[idx,0]).replace("_lr.tif",".bmp"))
        image = rgb2gray(image)
        label = rgb2gray(label)
        label = label / 255.0
        label = label.astype('float32')
        image = image / 255.0 # Normalizing [0;1]
        image = image.astype('float32') # Converting images to float32 
        
        return {'image': image, 'label': label}
