#!/usr/bin/env python
# coding: utf-8

# # 1 Getting started

# In[1]:


#%matplotlib notebook
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# # 2 Data Loader

# In[3]:


# Part 1
dataset_root_dir = '/datasets/ee285f-public/caltech_ucsd_birds/'


# In[4]:


#Part 2
class BirdsDataset(td.Dataset):
    def __init__(self, root_dir, mode="train", image_size=(224, 224)):
        super(BirdsDataset, self).__init__()
        self.image_size = image_size
        self.mode = mode
        self.data = pd.read_csv(os.path.join(root_dir, "%s.csv" % mode))
        self.images_dir = os.path.join(root_dir, "CUB_200_2011/images")
    def __len__(self):
        return len(self.data)
    def __repr__(self):
        return "BirdsDataset(mode={}, image_size={})".             format(self.mode, self.image_size)
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir,                                 self.data.iloc[idx]['file_path'])
        bbox = self.data.iloc[idx][['x1', 'y1', 'x2', 'y2']]
        img = Image.open(img_path).convert('RGB')
        img = img.crop([bbox[0], bbox[1], bbox[2], bbox[3]])
        b = 1
        a = -1
        maxmin = img.getextrema()
        minn = np.asarray([i[0] for i in maxmin])
        maxx = np.asarray([i[1] for i in maxmin])
        sigma = (maxx + minn)/2 
        mu    = (maxx-minn)/2
        
        transform = tv.transforms.Compose([
            # COMPLETE
            tv.transforms.Resize(self.image_size), 
            tv.transforms.ToTensor(), #Normalizes the image 
            tv.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
            ])
        
        x = transform(img)
        d = self.data.iloc[idx]['class']
        return x, d
    def number_of_classes(self):
        return self.data['class'].max() + 1


# In[5]:


#Part 3
def myimshow(image, ax=plt):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h


# In[6]:


train_set = BirdsDataset(dataset_root_dir)

x, d = train_set[10]

myimshow(x)


# In[7]:


# Part 4

train_loader = td.DataLoader(train_set, batch_size=16, shuffle=True,
                             pin_memory=True,)
print(type(train_loader))


# The advantage of using pin_memory= True is that loading samples on CPU and pushing it during training to the GPU, we can speed up the host to device transfer by enabling pin_memory. This lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer process. <br>
# There are 743/16~= 47 batches
# 

# In[8]:


# Part 5
for i,k in enumerate(train_loader):
    if i < 4:
        print(type(k[0]))
        f = k[0]
        print((k[1][0]))        
        plt.figure()
        myimshow(f[0])


# I have re-evaluated the cell, I have observed that each time we get different results: images and labels. <br>
# This happens because we have reshuffle the data each time and sampled different samples

# In[9]:


# Part 6
val_set = BirdsDataset(dataset_root_dir, mode='val')
val_loader = td.DataLoader(train_set, batch_size=16, shuffle=False,
                             pin_memory=True,)


# I believe that we need to shuffle the data set for training because we would like to change the batches at each iteration. <br>
# Shuffling the dataset each time adds randomness and diversity to the gradient descent and therefore makes it easier to converge to the minimum. <br>
# We shouldn't shuffle for validation because each time we go through the dataset for validation we need to make sure that we're validating on the same dataset which is not shuffled in order to evaluate the performance.
# 

# In[10]:


#!ln -s /datasets/ee285f-public/nntools.py
#m ./nntools.py


# In[11]:


import nntools as nt


# In[12]:


#Part 7

#help(nt.NeuralNetwork)
#net = nt.NeuralNetwork()


# When I tried: net = nt.NeuralNetwork(),
# the class couldn't be instantiated since it's abstract

# In[13]:


class NNClassifier(nt.NeuralNetwork):
    def __init__(self):
        super(NNClassifier, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    def criterion(self, y, d):
        return self.cross_entropy(y, d)


# # 4 VGG-16 Transfer Learning

# In[14]:


#Part 8
vgg = tv.models.vgg16_bn(pretrained=True)


# In[15]:


print(vgg)


# 
# The learnable parameters are the filters in the convulotions and the linear FC linear layers: <br>
# (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) <br>
# (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
# (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
# (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
# (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
# (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
# (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
# (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
# (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
# (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
# (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
# (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
# (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))<br>
# 
# (0): Linear(in_features=25088, out_features=4096, bias=True) <br>
# (3): Linear(in_features=4096, out_features=4096, bias=True) <br>
# (6): Linear(in_features=4096, out_features=1000, bias=True) <br>
# 

# In[16]:


#Part 9
class VGG16Transfer(NNClassifier):
    def __init__(self, num_classes, fine_tuning=False):
        super(VGG16Transfer, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad = fine_tuning
        self.features = vgg.features
        # COMPLETE
        self.classifier = vgg.classifier
        num_ftrs = vgg.classifier[6].in_features
        self.classifier[6] = nn.Linear(num_ftrs, num_classes) # required_grad is True for this layer

    def forward(self, x):
        # COMPLETE
        f = self.features(x)
        f1= f.view(16,-1)
        y = self.classifier(f1)
        return y
    


# 

# In[17]:


#Part 10
num_classes = train_set.number_of_classes()
net = VGG16Transfer(num_classes = num_classes)
print('The learnable parameters are: ')
for name, param  in net.named_parameters():
    print(name, param.size(), param.requires_grad)


# # 5 Training experiment and checkpoints

# In[18]:


#Part 11
class ClassificationStatsManager(nt.StatsManager):
    def __init__(self):
        super(ClassificationStatsManager, self).__init__()
    def init(self):
        super(ClassificationStatsManager, self).init()
        self.running_accuracy = 0
    def accumulate(self, loss, x, y, d):
        super(ClassificationStatsManager, self).accumulate(loss, x, y, d)
        _, l = torch.max(y, 1)
        self.running_accuracy += torch.mean((l == d).float())
    def summarize(self):
        loss = super(ClassificationStatsManager, self).summarize()
        accuracy = (100 * self.running_accuracy)/self.number_update # COMPLETE
        return {'loss': loss, 'accuracy': accuracy}


# #Part 12 <br>
# We use it to evaluate the experiment which forward propagates the validation set <br>
# through the network and returns the statistics computed by the stats manager <br>
# This method allows us to evaluate the current model by validating the parameters of the current experiment <br>
# This evaluation allows us to valuate the parameters and the model's perfromance on a dataset different than the training set

# In[19]:


#Part 13
lr = 1e-3
net = VGG16Transfer(num_classes)
net = net.to(device)
adam = torch.optim.Adam(net.parameters(), lr=lr)
stats_manager = ClassificationStatsManager()
exp1 = nt.Experiment(net, train_set, val_set, adam, stats_manager,
                    output_dir="birdclass1", perform_validation_during_training=True)


# A new "birdclass1" directory has been created <br>
# Two files has been created in that directory: config.txt and checkpoint.pth.tar <br> 
# I have visualized config.txt as requested, it is describing the setting of the experiment and saves the variable's values <br>
# checkpoint.pth.tar is a binary file containing the state of the experiment and a documentation of the experiment <br>

# In[20]:


# Part 14
lr = 1e-3

adam = torch.optim.Adam(net.parameters(), lr=lr)
stats_manager = ClassificationStatsManager()
exp1 = nt.Experiment(net, train_set, val_set, adam, stats_manager,
                        output_dir="birdclass1", perform_validation_during_training=True)


# Once the learning rate was changed from 1e-3 to 1e-4, I have got the following error: <br> 
#     ValueError: Cannot create this experiment: I found a checkpoint conflicting with the current setting. <Br>
# But once the learning rate was changed back to 1e-3, we didn't get the error. Appearantly, according to <br>
# Experiment class documentation, the last checkpoint was supposed to be uploaded, but because the settings <br>
# didn't match, and exception was raised and we got an error.
# 

# In[21]:


#Part 15
def plot(exp, fig, axes):
    axes[0].clear()
    axes[1].clear()
    axes[0].plot([exp.history[k][0]['loss'] for k in range(exp.epoch)],
    label="traininng loss")
    # COMPLETE
    axes[1].plot([exp.history[k][0]['accuracy'] for k in range(exp.epoch)],
    label="training accuracy")    
    #############################################################
    axes[0].plot([exp.history[k][1]['loss'] for k in range(exp.epoch)],
    label="evaluation loss")
    # COMPLETE
    axes[1].plot([exp.history[k][1]['accuracy'] for k in range(exp.epoch)],
    label="evaluation accuracy")    
    ################33

    axes[0].legend()
    axes[1].legend()
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy")
        
    
    plt.tight_layout()
    fig.canvas.draw()
    
fig, axes = plt.subplots(ncols=2, figsize=(7, 3))
exp1.run(num_epochs=20, plot=lambda exp: plot(exp, fig=fig, axes=axes))


# In[22]:


#Part 16
resnet = tv.models.resnet18(pretrained=True)
print(resnet)


# In[23]:



class Resnet18Transfer(NNClassifier):
    def __init__(self, num_classes, fine_tuning=False):
        super(Resnet18Transfer, self).__init__()
        
        for param in resnet.parameters():
            param.requires_grad = fine_tuning
        self.model = resnet
        # COMPLETE
        num_ftrs = list(resnet.children())[-1].in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        y = self.model(x)
        return y


# In[24]:


#Part 17
num_classes = train_set.number_of_classes()
print(num_classes)
net = Resnet18Transfer(num_classes = num_classes)
net = net.to(device)

print(net.device)

adam = torch.optim.Adam(net.parameters(), lr=lr)
stats_manager = ClassificationStatsManager()
exp2 = nt.Experiment(net, train_set, val_set, adam, stats_manager,
                        output_dir="birdclass3", perform_validation_during_training=True)


# In[25]:


def plot(exp, fig, axes):
    axes[0].clear()
    axes[1].clear()
    axes[0].plot([exp.history[k][0]['loss'] for k in range(exp.epoch)],
    label="traininng loss")
    # COMPLETE
    axes[1].plot([exp.history[k][0]['accuracy'] for k in range(exp.epoch)],
    label="training accuracy")    
    
    axes[0].plot([exp.history[k][1]['loss'] for k in range(exp.epoch)],
    label="evaluation loss")
    # COMPLETE
    axes[1].plot([exp.history[k][1]['accuracy'] for k in range(exp.epoch)],
    label="evaluation accuracy")        
    
    axes[0].legend()
    axes[1].legend()
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Accuracy")
    
    plt.tight_layout()
    fig.canvas.draw()
    
fig, axes = plt.subplots(ncols=2, figsize=(7, 3))
exp2.run(num_epochs=20, plot=lambda exp: plot(exp, fig=fig, axes=axes))


# In[26]:


#Part 18



