#!/usr/bin/env python
# coding: utf-8

# # Data paths to csv file



import os
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import PIL.Image as Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR
import torch.optim as optim

def generate_csv(Train_RGB_path,Train_Depth_path,Out_directory):
    df = pd.DataFrame(columns = ['RGB', 'Depth', 'Labels'])
    for count, filename in enumerate(os.listdir(Train_Depth_path)):
        K=os.path.join(Train_Depth_path,filename)
        L=os.path.join(Train_RGB_path,filename)
        df.loc[count, 'Depth'] = K
        df.loc[count, 'RGB'] = L
        AV=re.findall(r'-?\d*\.{0,1}\d+', filename)[2]

        df.loc[count, 'Labels'] = AV
    df.to_csv(Out_directory,index=False)
    return(df)
    

class loma_data(Dataset):
    
    """This is my own dataset class for data loader to fetch RGB image with its corresponding Depth image and  label"""
    
    def __init__(self,csv_path,rgb_img_directory,depth_img_dir,transform_rgb=None,transform_depth=None):
        
        """Initializing image directories, image paths to get image names from and output labels and also the transformation
        that's going to be applied on images"""
        
        df = pd.read_csv(csv_path) #reading training-images csv file that contains paths to both RGB, Depth images
        #### RGB,DEPTH image directories#########
        self.rgb_img_dir = rgb_img_directory
        self.depth_img_dir = depth_img_dir
        ##########Extracted paths from df#########
        self.all_imgs_rgb = os.listdir(rgb_img_directory)
        self.all_imgs_depth = os.listdir(depth_img_dir)
        
        self.rgb_paths=df['RGB'].values
        self.depth_paths=df['Depth'].values
        #######Labels###########
        self.y = df['Labels'].values
        ##Apply this transformation #######
        self.transform_rgb    = transform_rgb
        self.transform_depth  =transform_depth

    def __getitem__(self, index):
        img_rgb = Image.open(os.path.join(self.rgb_img_dir,self.all_imgs_rgb[index]))
        
        img_depth = Image.open(os.path.join(self.depth_img_dir,self.all_imgs_depth[index]))
        
        if self.transform_rgb is not None:
            img_rgb = self.transform_rgb(img_rgb)
            img_depth = self.transform_depth(img_depth)
        
        label = self.y[index]
        return img_rgb, img_depth,label

    def __len__(self):
                               
        """Returns the length of the dataset"""
                               
        return self.y.shape[0]


#### Creating dataloader for training and testing data ####
# tensor([0.5326, 0.5094, 0.5153]) tensor([3184.9036]) tensor([0.1806, 0.2050, 0.1970]) tensor([4051.8313])


def load_data():
    transform_RGB = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((240,240)),
                                        transforms.Normalize((0.5326, 0.5094, 0.5153),(0.1806, 0.2050, 0.1970)), ##mean=0.5,std=0.5 x=(x-mean)/std                               
                                        #transforms.Normalize((0.5542, 0.5240, 0.5316),(0.1458, 0.1961, 0.1814)),
                                       ])
    transform_Depth = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((240,240)),
                                        transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32))),

                                        transforms.Normalize((3184.9036),(4051.8313)),
                                        #transforms.Normalize((2401.2874),(2989.7883)),
                                       ])

    train_dataset = loma_data(csv_path=r'D:\Lamia\Clened\Train\path_to_images.csv',
                              rgb_img_directory=r'D:\Lamia\Clened\Train\RGB',
                              depth_img_dir=r'D:\Lamia\Clened\Train\Depth',
                              transform_rgb=transform_RGB,
                              transform_depth=transform_Depth)

    test_dataset = loma_data(csv_path=r'D:\Lamia\Clened\Test\path_to_images.csv' ,
                              rgb_img_directory=r'D:\Lamia\Clened\Test\RGB',
                              depth_img_dir=r'D:\Lamia\Clened\Test\Depth',
                              transform_rgb=transform_RGB,
                              transform_depth=transform_Depth)

    valid_dataset =loma_data(csv_path=r'D:\Lamia\Clened\Valid\path_to_images.csv' ,
                              rgb_img_directory=r'D:\Lamia\Clened\Valid\RGB',
                              depth_img_dir=r'D:\Lamia\Clened\Valid\Depth',
                              transform_rgb=transform_RGB,
                              transform_depth=transform_Depth)


    train_loader = DataLoader(train_dataset, batch_size=10,shuffle=True)
    print("Length of the train_loader:", len(train_loader)) #number of batches  (Whole samples/batch size)
    print("Size of trainloader sampler",len(train_loader.sampler))
    
    test_loader = DataLoader(test_dataset, batch_size=5)
    print("Length of the test_loader:", len(test_loader)) #number of batches  (Whole samples/batch size)
    print("Size of TestLoader sampler",len(test_loader.sampler))

    valid_loader = DataLoader(valid_dataset, batch_size=5,shuffle=True)
    print("Length of the valid_loader:", len(valid_loader)) #number of batches  (Whole samples/batch size)
    print("Size of ValidLoader sampler", len(valid_loader.sampler))

    return train_loader,test_loader,valid_loader




    
class Net2(nn.Module):
    
    def __init__(self):
        super(Net2,self).__init__()
        # convolutional layer (sees 240x240x3 image tensor)
        self.conv1 = nn.Conv2d(3,16,3,padding='same')        #output=(240-3+2)/1+1= 16*240*240
        # convolutional layer (sees 16*240*240 tensor)
        self.conv2 = nn.Conv2d(16, 16, 3, padding='same')  #output=16*240*240
        #########maxpool########################
        # convolutional layer (sees 16*120*120 image tensor)
        self.conv3 = nn.Conv2d(16,32,3,padding='same')        #output=(120-3+2)/1+1= 32*120*120
        # convolutional layer (sees 32*120*120 tensor)
        self.conv4 = nn.Conv2d(32, 32, 3, padding='same')  #output=32*120*120
        ########maxpool#########################
        # convolutional layer (sees 32*60*60 image tensor)
        self.conv5=nn.Conv2d(32,48,3,padding='same')        #output=(60-3+2)/1+1= 48*60*60
        # convolutional layer (sees 48*60*60 tensor)
        self.conv6 = nn.Conv2d(48, 48, 3, padding=1)  #output=48*60*60
        ########maxpool########################
        # convolutional layer (sees 48*30*30 image tensor)
        self.conv7 =nn.Conv2d(48,64,3,padding='same')        #output=(30-3+2)/1+1= 64*30*30
        # convolutional layer (sees 64*30*30 tensor)
        self.conv8 = nn.Conv2d(64, 64, 3, padding='same')  #output=64*30*30
        ########maxpool########################     
        ############################################
        self.pool = nn.MaxPool2d(2, 2) #output=64*15*15=14400
        ##############################################
        self.fc1 = nn.Linear(14400*2, 2880)
        self.fc2 = nn.Linear(2880, 288)
        self.fc3 = nn.Linear(288, 32)
        self.fc4 = nn.Linear(32, 1)
        ##############################################
        self.dropout = nn.Dropout(0.25)
        #############################################
        self.conv9 = nn.Conv2d(1,16,3,padding='same') 
        
        
    def forward(self,x1,x2):
        
        # add sequence of convolutional and max pooling layers
        x1 =  F.relu(self.conv1(x1))                           #output= 16*240*240
        x1=   self.pool(F.relu(self.conv2(x1)))                #output= 16*120*120
        x1 =  F.relu(self.conv3(x1))                           #output= 32*120*120
        x1 =  self.pool(F.relu(self.conv4(x1)))                #output= 32*60*60
        x1 =  F.relu(self.conv5(x1))                           #output= 48*60*60
        x1 =  self.pool(F.relu(self.conv6(x1)))                #output= 48*30*30
        x1 =  F.relu(self.conv7(x1))                           #output= 64*30*30
        x1 =  self.pool(F.relu(self.conv8(x1)))                #output= 64*15*15      
        x1 =  x1.view(-1,64*15*15)                             # flatten image input
        #self.dropout = nn.Dropout(0.25)
        #x1 =  F.relu(self.fc1(x1))                             #output = 1x1440
        #self.dropout = nn.Dropout(0.25)
        
        x2 =  F.relu(self.conv9(x2))                           #output= 16*240*240
        x2 =  self.pool(F.relu(self.conv2(x2)))                #output= 16*120*120
        x2 =  F.relu(self.conv3(x2))                           #output= 32*120*120
        x2 =  self.pool(F.relu(self.conv4(x2)))                #output= 32*60*60
        x2 =  F.relu(self.conv5(x2))                           #output= 48*60*60
        x2 =  self.pool(F.relu(self.conv6(x2)))                #output= 48*30*30
        x2 =  F.relu(self.conv7(x2))                           #output= 64*30*30
        x2 =  self.pool(F.relu(self.conv8(x2)))                #output=64*15*15 
        x2 =  x2.view(-1,64*15*15)                             # flatten image input
        #x2 =  F.relu(self.fc1(x2))                             #output=1x1440
        
        x =  torch.cat((x1, x2), dim=1)                        #output=14400*2
        x =  F.relu(self.fc1(x))                               #output=2880
        x =  F.relu(self.fc2(x))                               #output=288
        x =  F.relu(self.fc3(x))                               #output=32
        x =  self.fc4(x)                                       #output=1
                                    
        return x
    
class Net3(nn.Module):
    
    def __init__(self):
        super(Net3,self).__init__()
        # convolutional layer (sees 240x240x3 image tensor)
        self.conv1 = nn.Conv2d(3,16,3,padding='same')        #output=(240-3+2)/1+1= 16*240*240
        # convolutional layer (sees 16*240*240 tensor)
        self.conv2 = nn.Conv2d(16, 16, 3, padding='same')    #output=16*240*240
        #########maxpool########################
        # convolutional layer (sees 16*120*120 image tensor)
        self.conv3 = nn.Conv2d(16,32,3,padding='same')       #output=(120-3+2)/1+1= 32*120*120
        # convolutional layer (sees 32*120*120 tensor)
        self.conv4 = nn.Conv2d(32, 32, 3, padding='same')    #output=32*120*120
        ########maxpool#########################
        # convolutional layer (sees 32*60*60 image tensor)
        self.conv5=nn.Conv2d(32,48,3,padding='same')         #output=(60-3+2)/1+1= 48*60*60
        # convolutional layer (sees 48*60*60 tensor)
        self.conv6 = nn.Conv2d(48, 48, 3, padding=1)         #output=48*60*60
        ########maxpool########################
        # convolutional layer (sees 48*30*30 image tensor)
        self.conv7 =nn.Conv2d(48,64,3,padding='same')        #output=(30-3+2)/1+1= 64*30*30
        # convolutional layer (sees 64*30*30 tensor)
        self.conv8 = nn.Conv2d(64, 64, 3, padding='same')    #output=64*30*30
        ########maxpool########################     
        ############################################
        self.pool = nn.MaxPool2d(2, 2) #output=64*15*15=14400
        ##############################################
        self.fc1 = nn.Linear(14400, 1440)
        self.fc2 = nn.Linear(2880, 64)
        self.fc3 = nn.Linear(64, 2)                            #split into two weights
        self.fc4 = nn.Linear(1440, 720)
        self.fc5 = nn.Linear(720, 32)
        self.fc6 = nn.Linear(32, 1)
        ##############################################
        #self.dropout = nn.Dropout(0.25)
        #############################################
        self.conv9 = nn.Conv2d(1,16,3,padding='same') 
        
        
    def forward(self,x1,x2):
        
        # add sequence of convolutional and max pooling layers
        x1 =  F.relu(self.conv1(x1))                           #output= 16*240*240
        x1=   self.pool(F.relu(self.conv2(x1)))                #output= 16*120*120
        x1 =  F.relu(self.conv3(x1))                           #output= 32*120*120
        x1 =  self.pool(F.relu(self.conv4(x1)))                #output= 32*60*60
        x1 =  F.relu(self.conv5(x1))                           #output= 48*60*60
        x1 =  self.pool(F.relu(self.conv6(x1)))                #output= 48*30*30
        x1 =  F.relu(self.conv7(x1))                           #output= 64*30*30
        x1 =  self.pool(F.relu(self.conv8(x1)))                #output= 64*15*15      
        x1 =  x1.view(-1,64*15*15)                             # flatten image input
        #self.dropout = nn.Dropout(0.25)
        x1 =  F.relu(self.fc1(x1))                             #output = 1x1440
        #self.dropout = nn.Dropout(0.25)
        
        x2 =  F.relu(self.conv9(x2))                           #output= 16*240*240
        x2 =  self.pool(F.relu(self.conv2(x2)))                #output= 16*120*120
        x2 =  F.relu(self.conv3(x2))                           #output= 32*120*120
        x2 =  self.pool(F.relu(self.conv4(x2)))                #output= 32*60*60
        x2 =  F.relu(self.conv5(x2))                           #output= 48*60*60
        x2 =  self.pool(F.relu(self.conv6(x2)))                #output= 48*30*30
        x2 =  F.relu(self.conv7(x2))                           #output= 64*30*30
        x2 =  self.pool(F.relu(self.conv8(x2)))                #output=64*15*15 
        x2 =  x2.view(-1,64*15*15)                             # flatten image input
        x2 =  F.relu(self.fc1(x2))                             #output=1x1440
        #self.dropout = nn.Dropout(0.25)

        x =  torch.cat((x1, x2), dim=1)                        #output=1440*2=2880
        x =  F.relu(self.fc2(x))                               #output=64
        x =  F.relu(self.fc3(x))                               #output=2
        x =  (x[0,0] * x1)  +  (x[0,1] * x2)                   #output=1440
        x =  F.relu(self.fc4(x))                               #output=720
        x =  F.relu(self.fc5(x))                               #output=32
        x =  self.fc6(x)                                       #output=1
                                      
        return x

class Net1(nn.Module):
    
    def __init__(self):
        super(Net1,self).__init__()
        # convolutional layer (sees 240x240x3 image tensor)
        self.conv1 = nn.Conv2d(3,16,3,padding='same')        #output=(240-3+2)/1+1= 16*240*240
        # convolutional layer (sees 16*240*240 tensor)
        self.conv2 = nn.Conv2d(16, 16, 3, padding='same')  #output=16*240*240
        #########maxpool########################
        # convolutional layer (sees 16*120*120 image tensor)
        self.conv3 = nn.Conv2d(16,32,3,padding='same')        #output=(120-3+2)/1+1= 32*120*120
        # convolutional layer (sees 32*120*120 tensor)
        self.conv4 = nn.Conv2d(32, 32, 3, padding='same')  #output=32*120*120
        ########maxpool#########################
        # convolutional layer (sees 32*60*60 image tensor)
        self.conv5=nn.Conv2d(32,48,3,padding='same')        #output=(60-3+2)/1+1= 48*60*60
        # convolutional layer (sees 48*60*60 tensor)
        self.conv6 = nn.Conv2d(48, 48, 3, padding=1)  #output=48*60*60
        ########maxpool########################
        # convolutional layer (sees 48*30*30 image tensor)
        
        self.conv7 =nn.Conv2d(48,64,3,padding='same')        #output=(30-3+2)/1+1= 64*30*30
        # convolutional layer (sees 64*30*30 tensor)
        self.conv8 = nn.Conv2d(64, 64, 3, padding='same')  #output=64*30*30
        ########maxpool########################     
        ############################################
        self.pool = nn.MaxPool2d(2, 2) #output=64*15*15=14400
        ##############################################
        self.fc1 = nn.Linear(64*15*15, 1440)
        self.fc2 = nn.Linear(2880, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)                          #--> regression
        ##############################################
        self.dropout = nn.Dropout(0.25)
        #############################################
        self.conv9 = nn.Conv2d(1,16,3,padding='same') 
        
        
    def forward(self,x1,x2):
        
        # add sequence of convolutional and max pooling layers
        x1 =  F.relu(self.conv1(x1))                           #output= 16*240*240
        x1=   self.pool(F.relu(self.conv2(x1)))                #output= 16*120*120
        x1 =  F.relu(self.conv3(x1))                           #output= 32*120*120
        x1 =  self.pool(F.relu(self.conv4(x1)))                #output= 32*60*60
        x1 =  F.relu(self.conv5(x1))                           #output= 48*60*60
        x1 =  self.pool(F.relu(self.conv6(x1)))                #output= 48*30*30
        x1 =  F.relu(self.conv7(x1))                           #output= 64*30*30
        x1 =  self.pool(F.relu(self.conv8(x1)))                #output= 64*15*15      
        x1 =  x1.view(-1,64*15*15)                             # flatten image input
        x1 =  F.relu(self.fc1(x1))                             #output = 1x1440
        
        x2 =  F.relu(self.conv9(x2))                           #output= 16*240*240
        x2 =  self.pool(F.relu(self.conv2(x2)))                #output= 16*120*120
        x2 =  F.relu(self.conv3(x2))                           #output= 32*120*120
        x2 =  self.pool(F.relu(self.conv4(x2)))                #output= 32*60*60
        x2 =  F.relu(self.conv5(x2))                           #output= 48*60*60
        x2 =  self.pool(F.relu(self.conv6(x2)))                #output= 48*30*30
        x2 =  F.relu(self.conv7(x2))                           #output= 64*30*30
        x2 =  self.pool(F.relu(self.conv8(x2)))                #output=64*15*15 
        x2 =  x2.view(-1,64*15*15)                             # flatten image input
        x2 =  F.relu(self.fc1(x2))                             #output=1x1440
        
        x =  torch.cat((x1, x2), dim=1)                        #output=2880=1440+1440
        x =  F.relu(self.fc2(x))                               #output=64
        x =  F.relu(self.fc3(x))                               #output=32
        x =  self.fc4(x)                                       #output=1
        
        return x