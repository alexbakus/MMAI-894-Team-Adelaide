# MMAI-894-Team-Adelaide
Deep Learning Team Project

- [MMAI-894-Team-Adelaide](#mmai-894-team-adelaide)
  * [Data Preparation](#data-preparation)
    + [Dataset Source](#dataset-source)
    + [Creating  Classes Base on File Names](#creating--classes-base-on-file-names)
    + [Data Cleanning](#data-cleanning)
    + [Final Data For ML Input](#final-data-for-ml-input)
    + [Train Test Split](#train-test-split)
  * [Experimnentation](#experimnentation)
    + [‘Grid Searching’ Best Models](#-grid-searching--best-models)
  * [Model Trainning](#model-trainning)
    + [Hardware Acceleration](#hardware-acceleration)
    + [Dynamically Create Model](#dynamically-create-model)
    + [ImageGenerator](#imagegenerator)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


## Data Preparation
### Dataset Source
The [original data](https://data.mendeley.com/datasets/rscbjbr9sj/) is collected from Mendeley, Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images.

This [raw data](RawData/) contains many datasets and we are using the 'NORMAL' and 'PNEUMONIA' sorted as 2 classes by the folder structure. 
```
RawData
│ 
└───NORMAL
│   │   personXXXX_normal_YYYY.JPEG
│   │   ...
│   
└───PNEUMONIA
    │   personXXXX_bactiria_YYYY.JPEG
    │   personXXXX_virus_YYYY.JPEG
    │   ...
```
### Creating  Classes Base on File Names
Within the 'PNEUMONIA' directory, the files are named as either 'personXXXX_bactiria_YYYY.JPEG' or 'personXXXX_virus_YYYY.JPEG'. We use the names to split the PNEUMONIA directory into 2 directory 'bacteria' and 'virus'. We also resized and images to 300x300 and center cropped the images with the script [CropMaxSquare.py](CropMaxSquare.py)
The script outputs a new directory named [center_crop_300](center_crop_300/)
```
center_crop_300
│ 
└───normal
│   │   personXXXX_normal_YYYY.JPEG
│   │   ...
│   
└───bacteria
|   │   personXXXX_virus_YYYY.JPEG
|   │   ...
│   
└───bacteria
    │   personXXXX_bactiria_YYYY.JPEG
    │   ...
```
### Data Cleanning
We noticed that a significant x-ray images consist man-made objects, such as medical equipments. To avoid potential data leakage, we mannually deleted. The remained data is saved in the directory named [center_crop_clean_300](center_crop_clean_300/)
![maxCropEffect](docs/screenshots/maxCrop.PNG)

### Final Data For ML Input
Once these preprocessing steps were completed, we proceeded with various experiment configurations of convolutional neural networks. The details of which will be discussed in the succeeding section.
![fileDistribution](docs/screenshots/filedistribution.PNG)

### Train Test Split
Common to many proper Machine Learning process, the pre-processed images are splinted at ratio of 70% for training, 20% for validation, and 10% for test evaluation. This process is done by the script [Creat3Splits.py](Creat3Splits.py)

## Experimnentation
### ‘Grid Searching’ Best Models
As training a CNN can be computationally expensive, and grid search method of hyper parameter tuning is also very computationally expensive, with limited time in mind, the team have chosen the following commonly tuned hyper parameters:

Hyper parameters | Values Searched 
---------------- | ---------------
Number of Convolution Layers | 3, 4, 5
First Convolution Layer Size (subsequent layers are multiples of 2’s of the first layer and their layer number-1) | 16, 32
Number of Densely connected layers (actual number of neurons per dense layer is either 256 or 512 ) | 0, 1, 2
Learning Rate | 0.001, 0.01
Kernel Size | 2, 3, 4
Dropout Rates | 0%, 25%, 50%
Batch Size | 32
Max Epochs | 15
Activation Function | Relu
Padding Method | Valid
Kernel Initializer | Random Uniform
Optimizer | Adam, SGD (Stochastic Gradient Descent)
Loss Function | Categorical Crossentropy
Training Patients | 2
Input Size | 200x200x3


## Model Trainning
With Grid Searching and Dynamically Model Generation in mind we use script [model_withGridSearch.py](3_Classes/model_withGridSearch.py)

### Hardware Acceleration
Since we are testing many model candidates, the team decides to utilize GPU to accelerate the training process. In this case a Nvidia GTX1080 was used.
Windows enviroment is set up by following the [Guide](https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781) from towardsdatascience.com and the [instructions](https://www.pugetsystems.com/labs/hpc/The-Best-Way-to-Install-TensorFlow-with-GPU-Support-on-Windows-10-Without-Installing-CUDA-1187/) from pugetsystems.com and last, the Package [keras-gpu](https://anaconda.org/anaconda/keras-gpu) was installed to enable the use of GPU in the training process.

### Dynamically Create Model
With all the tunable parameters, a total of 648 possible combinations of the CNN setup can be created and evaluated.
Each model was generated dynamically base on the current hyperparameter combination.

<table>
<tr>
  <th>Relavent Code</th>
  <th>Annotations</th>
</tr>
<tr>
<td>
    
  ```python
  model = Sequential()
```
</td>
<td>Initialize Model<br></td></tr>
<tr>
<td>

  ```python
model.add(Conv2D(layer_size,
        kernel_size=(kernel_size, kernel_size),
        activation='relu',
        padding='valid',
        kernel_initializer='random_uniform',
        input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_rate))
```
</td>
<td>Input Layer</td></tr>
<tr>
<td>

  ```python
for l in range(conv_layer - 1):
model.add(Conv2D(layer_size*(2**l),
                 kernel_size=(kernel_size, kernel_size),
                 activation='relu',
                 padding='valid',
                 kernel_initializer='random_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_rate))
```
</td>
<td>Hidden Convolution Layers 2,3,4,5<br></td></tr>
<tr>
<td>

  ```python
model.add(Flatten())
```
</td>
<td>Flatten</td></tr>
<tr>
<td>

  ```python
for _ in range(dense_layer):
    model.add(Dense(layer_size*16))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
```
</td>
<td>Dense Layer none,1,2</td></tr>
<tr>
<td>

  ```python
model.add(Dense(3, activation='softmax'))
```
</td>
<td>Output Layer</td></tr>
<tr>
<td>

  ```python
model.compile(loss='categorical_crossentropy',
                optimizer = optimizer,
                lr = learn_rate,
                metrics=['accuracy'])
``` 
</td>
<td>Compile Model</td></tr>
</table>

### ImageGenerator
To have more training images, we implemented image augmentation with the ImageDatagenerator from tensorflow.keras.preprocessing.image with the following parameters:

Parameters | Values
---------- | ------
Rescale | 1./255
Shear Range | 0.2
Rotation Range | 40
Zoom Range | 0.2
Width Shift Range | 0.2
Height Shift Range | 0.2
Horizontal Flip | True


