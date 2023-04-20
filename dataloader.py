import torch
import os
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from torch.utils.data import Dataset, DataLoader
from skimage import transform
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class Datasets(Dataset):
    def __init__(self,image_dir, label_dir, opt):
        self.opt = opt
        self.imagename = sorted(os.listdir(image_dir))
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
        if self.upsample == True:
            image = transform.rescale(image,2)
            image = (image>0.5)*255
            mask_name = os.path.join(self.mask_dir,str(self.labels.iloc[idx,0]).replace("_lr.tif",".bmp"))
        
        label = label / 255.0
        label = label.astype('float32')
        image = image / 255.0 # Normalizing [0;1]
        image = image.astype('float32') # Converting images to float32

        p = random.random()
        rot = random.randint(-45,45)
        image,label=TF.to_pil_image(image),TF.to_pil_image(label)
        image,label=TF.rotate(image,rot),TF.rotate(label,rot)
        if p<0.3:
            image,label=TF.vflip(image),TF.vflip(label)
        p = random.random()
        if p<0.3:
            image,label=TF.hflip(image),TF.hflip(label)
        p = random.random()
        if p>0.2:
            image,label=TF.affine(image,angle=0,translate=(0.1,0.1),shear=0,scale=1),TF.affine(label,angle=0,translate=(0.1,0.1),shear=0,scale=1)
        image,label=TF.to_tensor(image),TF.to_tensor(label)
        
        return {'image': image, 'label': label, 'name': self.imagename[idx]}
