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
import dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--label_dir", default = "/gpfsstore/rech/tvs/uki75tv/Train_LR_segmented", help = "path to label csv file")
parser.add_argument("--image_dir", default = "/gpfsstore/rech/tvs/uki75tv/Train_Label_trab_100", help = "path to image directory")
parser.add_argument("--batch_size", default = 16, help = "Batch size")
parser.add_argument("--nb_epochs", default = 100, help = "epochs")
parser.add_argument("--nb_workers", default = 4, help = "worker")
parser.add_argument("--lr",default=1e-3,help="learning rate")
opt = parser.parse_args()

NB_DATA = 7100
index = range(NB_DATA)
split = train_test_split(index,test_size=1000,shuffle=False)
datasets = dataloader.Datasets(opt.image_dir,opt.label_dir,opt) # Create dataset
trainloader = DataLoader(datasets, batch_size = opt.batch_size, sampler = split[0], num_workers = opt.nb_workers )
testloader =DataLoader(datasets, batch_size = 1, sampler = split[1], num_workers = opt.nb_workers )

criterion = nn.BCEWithLogitsLoss()
model = Model.Unet()
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

for epoch in range(opt.nb_epochs):
  for i, data in enumerate(trainloader,0):
    labels, inputs = data['label'],data['image']
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backpropagation and optimizer step
    loss.backward()
    optimizer.step()
  with torch.no_grad():
    for i, data in enumerate(testloader,0):
      outputs=model(inputs)
      metric = iou_metric(outputs,labels)
    test_metric.append(metric)
    print("Epoch",epoch,": Metric=",mean(test_metric))
