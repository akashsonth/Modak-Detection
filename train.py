#Import Libraries
print("Importing libraries...")
import os, os.path
import numpy as np
import torch
import torch.nn as nn 
import torchvision
from torchvision import datasets, models, transforms
from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from PIL import Image
import pickle
print("Libraries imported")

#Directories
HOME_DIR = os.getcwd()
TRAINDATA_DIR = os.path.join(HOME_DIR, 'Data/Train')
VALDATA_DIR = os.path.join(HOME_DIR, 'Data/Val')
validation_flag = input("Do you want to fine-tune the model?: Enter y/n \n")

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


#Initialize arrays for training and tuning the SVM
train_x = np.zeros((1,4096))
train_y = np.zeros((1), dtype='int64')
val_x = np.zeros((1,4096))
val_y = np.zeros((1), dtype='int64')

print("Extracting features of training data...")
#Extract features for training data
n = len(os.listdir(TRAINDATA_DIR)) #Size of training data
for i, img_name in enumerate(os.listdir(TRAINDATA_DIR)):
    print("Progress - ", int(100*i/n), "%", end='\r')
    image = Image.open(os.path.join(TRAINDATA_DIR, img_name))
    image_t = transform(image)
    batch_t = torch.unsqueeze(image_t, 0)
    out = model_conv(batch_t)
    np_out = np.asarray(out)
    train_x = np.concatenate((train_x,np_out))
        
    if img_name[0] == 'p': 
        #Class 1
        train_y = np.concatenate((train_y, np.array([1])))
    else:
        #Class 0
        train_y = np.concatenate((train_y, np.array([0])))

#Extract features for validation data
if(validation_flag == 'y' or validation_flag == 'Y'):
    print("Extracting features of validation data...")
    labels = ['neg', 'pos']    
    m = (len(os.listdir(os.path.join(VALDATA_DIR, labels[0]))) 
         + len(os.listdir(os.path.join(VALDATA_DIR, labels[1])))) #Size of validation data

    j = 0 #Counter for printing progress
    for label in labels:    
        for img_name in os.listdir(os.path.join(VALDATA_DIR, label)):
            print("Progress - ", int(j*100/m), "%", end ="\r")
            j += 1
            image = Image.open(os.path.join(VALDATA_DIR, label, img_name))
            image_t = transform(image)
            batch_t = torch.unsqueeze(image_t, 0)
            out = model_conv(batch_t)
            np_out = np.asarray(out)
            val_x = np.concatenate((val_x,np_out))
            if label == 'neg':
                val_y = np.concatenate((val_y, np.array([0])))
            else:
                val_y = np.concatenate((val_y, np.array([1])))
       
print("Features extracted")
    

#Remove the initial row of zeros present due to initialization    
train_x = train_x[1:,:]   
train_y = train_y[1:]
val_x = val_x[1:,:]   
val_y = val_y[1:]

#SVM training
print("Training Linear SVM...")
svm_model_linear = LinearSVC(C = 1e-4, verbose=0, max_iter=5000).fit(train_x, train_y) 
print("Training complete")

#Save the classifier
with open('linearSVM.pkl', 'wb') as fid:
    pickle.dump(svm_model_linear, fid)  
print("Model Saved")

if(validation_flag == 'y' or validation_flag == 'Y'):
    #Testing on the validation data
    print("----------Validation Metrics----------")
    print("Accuracy =", svm_model_linear.score(val_x, val_y)) 
    svm_predictions = svm_model_linear.predict(val_x)
    test_cm = confusion_matrix(val_y, svm_predictions) 
    print("Confusion matrix =\n",test_cm)