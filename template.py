#!/usr/bin/env python2.7

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.models import resnet50
from torchvision.utils import save_image


# --------------------------------------------------------------------------------
#   Helper functions; do not edit!
# --------------------------------------------------------------------------------
def convert_to_tensor(filepath):
    img = Image.open(filepath)
    preprocess = transforms.Compose([
            transforms.Resize([224, 224]), 
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    return preprocess(img)[None, :, :, :]

def convert_to_unnormalized_tensor(path):
    img = Image.open(path)
    preprocess = transforms.Compose([
            transforms.Resize([224, 224]), 
            transforms.ToTensor()
        ])
    return preprocess(img)[None, :, :, :]

def normalize(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return (x - torch.Tensor(mean).type_as(x)[None,:,None,None]) / torch.Tensor(std).type_as(x)[None,:,None,None]



# load a pre-trained neural network
model = resnet50()
model.load_state_dict((torch.load('resnet50-19c8e357.pth')))
model.eval()

# base images
pig_data   = convert_to_tensor("pig.jpg")
plane_data = convert_to_tensor("plane.jpg")

# --------------------------------------------------------------------------------
#   Challenge I: Make the model misclassify the pig image to a plane
#   You have two tensors, containing the pig and a plane image data
#   Both have the shape of (1, 3, 244, 244)
#   Your job is to interpolate the two data and create a flying pig image!
#   (ex. data_for_fooling = plane_data * 0.5 + pig_data * 0.5)
# --------------------------------------------------------------------------------

# TODO: your code here
data_for_fooling = pig_data

# Do not edit: save the resulting image
save_image(data_for_fooling, 'flying_pig.jpg')



# --------------------------------------------------------------------------------
#   Challenge II: Make the model misclassify the pig image to a plane
#   You have two tensors, containing the pig and a plane image data
#   Both have the shape of (1, 3, 244, 244)
#   Now, let's use gradients to update the pig image
# --------------------------------------------------------------------------------

# base image
pig = convert_to_unnormalized_tensor("pig.jpg")

# perturbations
pert = torch.zeros_like(pig, requires_grad=True)
opt  = optim.SGD([pert], lr=5e-3)
eps  = 2./255                       # max perturbation is 2./255 in l-infinity norm

# run this attack procedure 20 times
for _ in tqdm(range(20), desc=' : [run attack]'):
    pig_pert = pig + pert 
    pig_pred = model(normalize(pig_pert))
    
    # TODO: your code here
    # (Tip: modify the loss function by choosing the correct class number; 
    #       341 is hog and 404 is airliner; - means getting far, + means getting close to the class)
    loss = (-nn.CrossEntropyLoss()(pig_pred, torch.LongTensor([0])) \
            + nn.CrossEntropyLoss()(pig_pred, torch.LongTensor([0])))

    # update the pig
    opt.zero_grad()
    loss.backward()
    opt.step()

    # post-process; bound the perturbation
    pert.data.clamp_(-eps, eps)


# save the perturbed dog into the pytorch tensor format
perturbed_pig = normalize(pig + pert)
torch.save(perturbed_pig, 'flying_pig.pt')
