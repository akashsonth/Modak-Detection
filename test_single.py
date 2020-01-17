#Import Libraries
print("Importing libraries...")
import os, os.path
import numpy as np
import math
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import cv2 as cv
from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from PIL import Image
import pickle
print("Libraries imported")

#Directories
HOME_DIR = os.getcwd()
img_name = os.path.join(HOME_DIR, 'Pictures/Test/momos.jpg')

#Similar transforms as ImageNet on which model has been trained
from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(224),                    #[2]
 transforms.ToTensor(),                     #[3]
 transforms.Normalize(                      #[4]
 mean=[0.485, 0.456, 0.406],                #[5]
 std=[0.229, 0.224, 0.225]                  #[6]
 )])


#Pretrained VGG19 without final softmax layer
model_conv = torchvision.models.vgg19(pretrained=True)
for param in model_conv.features.parameters():
    param.requires_grad = False
for param in model_conv.classifier.parameters():
    param.requires_grad = False
new_classifier = nn.Sequential(*list(model_conv.classifier.children())[:-1])
model_conv.classifier = new_classifier
model_conv.eval()

#Load the classifier
with open('linearSVM.pkl', 'rb') as fid:
    svm_model_linear = pickle.load(fid)

    image = Image.open(img_name)
    image_t = transform(image)
    batch_t = torch.unsqueeze(image_t, 0)
    out = model_conv(batch_t)
    test_x = np.asarray(out)

svm_predictions = svm_model_linear.predict(test_x)
print(svm_predictions)

