import torch
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import random
import Model
import pickle
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from sklearn.model_selection import train_test_split
from trainer import Trainer
import dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--label_dir", default = "./Test_Label_6p.csv", help = "path to label csv file")
parser.add_argument("--image_dir", default = "/gpfsstore/rech/tvs/uki75tv/MOUSE_BPNN/HR/Test_Label_trab", help = "path to image directory")
parser.add_argument("--batch_size", default = 16, help = "Batch size")
parser.add_argument("--nb_epochs", default = 100, help = "Batch size")
parser.add_argument("--lr",default=1e-3,help="learning rate")
opt = parser.parse_args()

NB_DATA = 4073
index = range(NB_DATA)
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
