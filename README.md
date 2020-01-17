# Object Recognition
Train your own 2 class object recognition model without writing any code.\
Alternatively, the provided model can also be used which has been trained to recognize whether an image is of a Modak (an Indian sweet) or not.

The first part of this README file comprises the methodology and results. \
The second part details the procedure to run the code .


## Method and Results-
1. Data Collection-
Images are collecting from Google images using the Chrome extension 'Fatkun Batch Download Image'.\
Alternatively, images can also be obtained using Microsoft's Bing Image Search API.\
All the images are resized to 224x224x3.

2. Augmentation-
The augmentation methods used in this project are grayscale conversion, adding speckle noise, flipping the image horizontally, and rotating the image by a random angle. Although these augmentation methods have been applied individually, multiple augmentations can also be applied on the same image. Other augmentation methods such as affine transformation, skew, shift, padding, and color transformation such as brightness can also be performed.

3. Model creation-
Since the dataset chosen is a very small one, training a CNN from scratch would lead to underfitting. Therefore, transfer learning was chosen to be done using VGG19 which was pre-trained on the ImageNet database. This was done using the PyTorch library since loading and modifying state-of-the-art architectures such as VGG19 can be done very conveniently. \
The final softmax layer is removed so that the model can be used for the required number of classes and on a dataset different from ImageNet. \
An SVM with a linear kernel is used for classifying the extracted features since SVMs are very good binary classifiers. The scikit-learn library is used for this purpose. Alternatively, an SVM can also be created as a fully connected layer with 'linear' activation and 'hinge' loss (https://stackoverflow.com/questions/54414392/convert-sklearn-svm-svc-classifier-to-keras-implementation).

4. Training and tuning the model-
Features for all the images are extracted from VGG19 and compiled. These are then fed to train the SVM. The hyperparameter C for SVM is varied from 100000 to 0.00001 in powers of 10. Optimum value of C is obtained as 0.0001 at which validation accuracy is maximum, i.e., 80%.
The test accuracy obtained at optimum value of C is 85%. (See images in the screenshots folder for confusion matrix)\
'Screenshots/Validation-- C 1e4.png'- Validation metrics for C = 10000 \
'Screenshots/Validation-- C 1e-3.png'- Validation metrics for C = 0.001 \
'Screenshots/Validation-- C 1e-4.png'- Validation metrics for C = 0.0001 \
'Screenshots/test.png'- Test metrics for C = 0.0001 

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


