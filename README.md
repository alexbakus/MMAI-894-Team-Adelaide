# MMAI-894-Team-Adelaide
Deep Learning Team Project

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
<table style="border-collapse:collapse;border-spacing:0;border-color:#ccc;table-layout: fixed; width: 710px" class="tg"><colgroup><col style="width: 431px"><col style="width: 279px"></colgroup><tr><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;text-align:right;vertical-align:top">Hyper parameters</th><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;text-align:left;vertical-align:top">Values Searched</th></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Number of Convolution Layers</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">3, 4, 5</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:right;vertical-align:top">First Convolution Layer Size (subsequent layers are multiples of 2’s of the first layer and their layer number-1)</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top"><br>16, 32</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Number of Densely connected layers (actual number of neurons per dense layer is either 256 or 512 )</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">0, 1, 2</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:right;vertical-align:top">Learning Rate</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">0.001, 0.01</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Kernel Size</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">2, 3, 4</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:right;vertical-align:top">Dropout Rates</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">0%, 25%, 50%</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Batch Size</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">32</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:right;vertical-align:top">Max Epochs</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">15</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Activation Function</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">Relu</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:right;vertical-align:top">Padding Method</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">Valid</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Kernel Initializer</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">Random Uniform</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:right;vertical-align:top">Optimizer</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">Adam, SGD (Stochastic Gradient Descent)</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Loss Function</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">Categorical Crossentropy</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:right;vertical-align:top">Training Patients</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">2</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Input Size</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">200x200x3</td></tr></table>
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

### ImageGenerators
To have more training images, we implemented image augmentation with the ImageDatagenerator from tensorflow.keras.preprocessing.image with the following parameters:
<table style="border-collapse:collapse;border-spacing:0;border-color:#ccc;table-layout: fixed; width: 710px" class="tg"><colgroup><col style="width: 431px"><col style="width: 279px"></colgroup><tr><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;text-align:right;vertical-align:top">Hyper parameters</th><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;text-align:left;vertical-align:top">Values Searched</th></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Number of Convolution Layers</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">3, 4, 5</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:right;vertical-align:top">First Convolution Layer Size (subsequent layers are multiples of 2’s of the first layer and their layer number-1)</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top"><br>16, 32</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Number of Densely connected layers (actual number of neurons per dense layer is either 256 or 512 )</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">0, 1, 2</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:right;vertical-align:top">Learning Rate</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">0.001, 0.01</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Kernel Size</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">2, 3, 4</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:right;vertical-align:top">Dropout Rates</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">0%, 25%, 50%</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Batch Size</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">32</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:right;vertical-align:top">Max Epochs</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">15</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Activation Function</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">Relu</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:right;vertical-align:top">Padding Method</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">Valid</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Kernel Initializer</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">Random Uniform</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:right;vertical-align:top">Optimizer</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">Adam, SGD (Stochastic Gradient Descent)</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Loss Function</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">Categorical Crossentropy</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:right;vertical-align:top">Training Patients</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">2</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:right;vertical-align:top">Input Size</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">200x200x3</td></tr></table>

#### .flow_from_directory
All of the training and validation images are augmented and generated on-demand with the method of ‘.flow_from_directory’ 
  ```python
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical')
``` 

  ```python
validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical')
``` 

### Training Progress Monitoring and Model Saving 
As the searching and training sequence is expected to run for about a week, it is important for the team to be able to view the current progress and visualizes past model performance through a monitoring platform. 

#### Tensorbaord Callback
TensorBoard is the tool to monitor the progress of each model’s training and validation performance. 
  ```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
``` 
#### EarlyStop Callback
To minimizing the chance of over Fitting, EarlyStopping method is implemented and continuing monitor the ‘val_loss’ with ‘patience’ set at 2 epochs.
  ```python
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1,patience=2)
``` 
#### CheckPoint Callback
To save the model with its best epoch weight, ModelCheckpoint method is implemented and each model is given a name that describes the hyperparameters set tagged by the current datetime to ensure uniqueness.
  ```python
checkpoint_callback =  tf.keras.callbacks.ModelCheckpoint(filepath=(model_export + mdoel_name+ model_time + '.h5'), monitor='val_accuracy',  save_best_only=True)
``` 
### .fit_generator
With the use of training and validation image generation method, ‘. fit_generator’ method is used as the method of fit for each model.
  ```python
model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    callbacks=[tensorboard_callback, es_callback, checkpoint_callback])
``` 

### Identify the Top Model
Top 10 models are manually selected base on their validation accuracy.
Top preforming models’ hyper parameter sets are collected, recreated and trained with 300x300 input size and evaluated on the test data set. These 10 models took an additional 10 hours to train and evaluate.
<table style="border-collapse:collapse;border-spacing:0;border-color:#ccc" class="tg"><tr><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;text-align:left;vertical-align:top">Name</th><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;text-align:left;vertical-align:top">Train Accuracy</th><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;text-align:left;vertical-align:top">Validation Accuracy</th><th style="font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f0f0f0;text-align:left;vertical-align:top">Test Accuracy</th></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#32cb00;text-align:left;vertical-align:top">5_conv_32_nodes_1_dense_4_kernelSz_adam_optimizer_0_001lr_0_0_dropr</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#32cb00;text-align:left;vertical-align:top">75.9%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#32cb00;text-align:left;vertical-align:top">75.2%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#32cb00;text-align:left;vertical-align:top">77.2%</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">4_conv_16_nodes_1_dense_2_kernelSz_adam_optimizer_0_01lr_0_0_dropr</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">73.8%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">73.4%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">74.8%</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">5_conv_16_nodes_1_dense_3_kernelSz_adam_optimizer_0_001lr_0_0_dropr</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">73.7%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">73.3%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">74.5%</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">4_conv_16_nodes_1_dense_3_kernelSz_adam_optimizer_0_01lr_0_0_dropr</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">73.5%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">72.6%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">74.2%</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">5_conv_32_nodes_0_dense_4_kernelSz_adam_optimizer_0_01lr_0_0_dropr</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">74.3%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">73.7%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">73.6%</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">4_conv_32_nodes_2_dense_4_kernelSz_adam_optimizer_0_001lr_0_0_dropr</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">71.0%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">72.1%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">70.9%</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">5_conv_32_nodes_0_dense_4_kernelSz_adam_optimizer_0_001lr_0_0_dropr</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">72.7%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">73.0%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">69.1%</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">5_conv_32_nodes_0_dense_2_kernelSz_adam_optimizer_0_001lr_0_0_dropr</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">70.5%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">69.6%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fe0000;text-align:left;vertical-align:top">68.5%</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">5_conv_32_nodes_1_dense_3_kernelSz_adam_optimizer_0_001lr_0_0_dropr</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f8a102;text-align:left;vertical-align:top">73.6%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#f9f9f9;text-align:left;vertical-align:top">68.8%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fe0000;text-align:left;vertical-align:top">68.5%</td></tr><tr><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">5_conv_16_nodes_1_dense_3_kernelSz_adam_optimizer_0_01lr_0_0_dropr</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">66.0%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fff;text-align:left;vertical-align:top">66.6%</td><td style="font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#ccc;color:#333;background-color:#fe0000;text-align:left;vertical-align:top">61.9%</td></tr></table>
Looking at the retrained results, descending sorted by their test accuracy, most of the models training results tracks very well towards its Validation accuracy to within 1% difference (highlighted in yellow if outside of the range). This shows that the early stop setting was effective at making sure that the training process is not dramatically overfitting.

On the other hand, looking at the bottom 4 models, their test accuracy differs their training accuracy my more than 1% (highlighted in red), this shows that despite the effort in cleaning, balancing the class samples and early stopping strategies, some level of overfitting still occurred and having the test split is always the best practice to evaluating the true model performance. 

### The Best Preforming Model
Based on all the experiments the team ran, as specified in the Experiments section 3, the CNN architecture that performed the best, given the computational resources and time, is provided below
