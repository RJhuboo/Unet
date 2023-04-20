import torch
import os
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
import random
import Model
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
import dataloader
import time
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--label_dir", default = "/gpfsstore/rech/tvs/uki75tv/Train_LR_segmented", help = "path to label csv file")
parser.add_argument("--image_dir", default = "/gpfsstore/rech/tvs/uki75tv/Train_Label_trab_100", help = "path to image directory")
parser.add_argument("--batch_size", default = 16,type=int, help = "Batch size")
parser.add_argument("--nb_epochs", default = 100,type=int, help = "epochs")
parser.add_argument("--nb_workers", default = 4,type=int, help = "worker")
parser.add_argument("--lr",default=5e-3,type=float,help="learning rate")
opt = parser.parse_args()

NB_DATA = 7100
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
index = range(NB_DATA)
index = []

start = 49
end = 99

while start <= 7099:
    for i in range(start, end+1):
        index.append(i)
    start += 100
    end += 100
split = train_test_split(index,test_size=500,shuffle=False)
datasets = dataloader.Datasets(opt.image_dir,opt.label_dir,opt) # Create dataset
trainloader = DataLoader(datasets, batch_size = opt.batch_size, sampler = split[0], num_workers = opt.nb_workers )
testloader =DataLoader(datasets, batch_size = 1, sampler = split[1], num_workers = opt.nb_workers )

criterion = nn.BCEWithLogitsLoss()
model = Model.UNet()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

# Assuming your U-Net model is called `model`
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

def iou_metric(outputs, targets, threshold=0.5):
    """
    Computes the Intersection over Union (IoU) metric for binary image segmentation.
    """
    outputs = (outputs > threshold).float()
    intersection = (outputs * targets).sum()
    union = (outputs + targets).sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou

test_metric = []
train_loss = []
for epoch in range(opt.nb_epochs):
    start_time= time.time()
    for i, data in enumerate(trainloader,0):
        labels, inputs = data['label'].to(device),data['image'].to(device)
        inputs = inputs.reshape(inputs.size(0),1,512,512)
        labels = labels.reshape(labels.size(0),1,512,512)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        train_loss.append(loss.item())
        # Backpropagation and optimizer step
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Running Loss: {:4f}".format(loss.item()))
    end_time = time.time()
    total_time = end_time - start_time
    print("Epoch {}, in {:2f}: Loss={:4f}".format(epoch,total_time,np.mean(train_loss)))
    
    with torch.no_grad():
        for i, data in enumerate(testloader,0):
            labels,inputs,imagename = data['label'].to(device),data['image'].to(device),data['name']
            inputs = inputs.reshape(inputs.size(0),1,512,512)
            labels = labels.reshape(labels.size(0),1,512,512)
            outputs=model(inputs)
            metric = iou_metric(outputs,labels)
            test_metric.append(metric.cpu())
            save_image(outputs[0],"./output2/"+imagename[0].replace(".png","_output.png"))
        print("Epoch",epoch,": Metric=",np.mean(test_metric))
