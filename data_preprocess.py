#Import Libraries
print("Importing libraries...")
import cv2 as cv
import os, os.path
import numpy as np
import random
import imutils
print("Libraries imported")

#Directories
HOME_DIR = os.getcwd()
PICS_DIR = os.path.join(HOME_DIR, 'Pictures/Train') #Unprocessed data
TRAINDATA_DIR = os.path.join(HOME_DIR, 'Data/Train')

#Image Augmentation
def dataAugment(img, label, count):    
    img = cv.resize(img, (224,224))
    cv.imwrite(os.path.join(TRAINDATA_DIR, label +'_'+ str(count) + '.png'), img)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #Convert RGB image to Grayscale
    img_gray_3ch = np.zeros_like(img)
    img_gray_3ch[:,:,0] = img_gray
    img_gray_3ch[:,:,1] = img_gray
    img_gray_3ch[:,:,2] = img_gray
    cv.imwrite(os.path.join(TRAINDATA_DIR, label +'_'+ str(count) + '_gray.png'), img_gray_3ch)

    #Add speckle noise
    row,col,ch = img.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)        
    img_noise = img + img * gauss
    cv.imwrite(os.path.join(TRAINDATA_DIR, label +'_'+ str(count) + '_noise.png'), img_noise)

    img_flip = cv.flip(img, 1) #Flip horizontally
    cv.imwrite(os.path.join(TRAINDATA_DIR, label +'_'+ str(count) + '_flip.png'), img_flip)
    
    angle = random.choice([-90,-75,-60,-45,-30,-15,15,30,45,60,75,90]) #Choose random angle from given angles
    img_rotate = imutils.rotate(img, angle)
    cv.imwrite(os.path.join(TRAINDATA_DIR, label +'_'+ str(count) + '_rot.png'), img_rotate)


def main():
    print("Processing images...")

    #Preprocess all the images of Class 0
    count = 0
    label = 'neg'
    for img_name in os.listdir(os.path.join(PICS_DIR, label)):
        count += 1
        img = cv.imread(os.path.join(PICS_DIR, label, img_name))
        dataAugment(img, label, count)

    #Preprocess all the images of Class 1
    count = 0
    label = 'pos'
    for img_name in os.listdir(os.path.join(PICS_DIR, label)):
        count += 1
        img = cv.imread(os.path.join(PICS_DIR, label, img_name))
        dataAugment(img, label, count)
    
    print("Processing completed")


if __name__ == "__main__":
    main()
