# Object Recognition
Train your own 2 class object recognition model without writing any code.\
Alternatively, the provided model can also be used which has been trained to recognize whether an image is of a Modak (an Indian sweet) or not.

The first part of this README file comprises the methodology and results. \
The second part details the procedure to run the code .


## Model-
The method of transfer learning has been used by using the VGG19 architecture trained pretrained on the ImageNet database. The final softmax layer in VGG19 has been removed. Instead, the 1x4096 output from VGG19 has been used as input features for an SVM with a linear kernel.\
(224x224x3 Image) -> VGG19 -> (1x4096 features) -> SVM -> 0/1 Classification


## To test the provided modak recognition model-
git clone https://github.com/akashsonth/Object-Recognition \
cd Object-Recognition

Run the python script 'test_single.py' in case single image is to be tested. A prompt will appear asking for the entire image path.\
Run the python scipt 'test_data.py' in case testing is to be done on images in the Test folder.


## To train your own model-

### Step 0:
git clone https://github.com/akashsonth/Object-Recognition \
cd Object-Recognition

### Step 1 (Data Collection): 
Collect train images of both classes (Class 0 and Class 1). Store the images for Class 0 in 'Pictures/neg', and the images for Class 1 in 'Pictures/pos'.

To test on multiple images together, collect test images of both classes. Store the images for Class 0 in 'Test/neg', and the images for Class 1 in 'Test/pos'.

In case SVM is to be fine-tuned later, collect validation images for both classes. Store the images for Class 0 in 'Val/neg', and the images for Class 1 in 'Val/pos'.

### Step 2 (Augmentation):
Run the python 'script data_preprocess.py'. A prompt will appear asking whether data augmentation is to be done or not.

### Step 3 (Training and tuning the model):
Run the python script 'train.py'. A prompt will appear asking whether fine-tuning is to be done or not. This helps tune the SVM model based on the validation metrics. Also rename the SVM model (which is saved in the form of a pickle file) to avoid rewriting the modak recognition model ('linearSVM.pkl').

### Step 4 (Testing the model):
Run the python script 'test_single.py' in case single image is to be tested. A prompt will appear asking for the entire image path.
Run the python scipt 'test_data.py' in case testing is to be done on images in the Test folder


