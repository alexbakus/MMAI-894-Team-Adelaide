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
Within the 'PNEUMONIA' directory, the files are named as either 'personXXXX_bactiria_YYYY.JPEG' or 'personXXXX_virus_YYYY.JPEG'. We use the names to split the PNEUMONIA directory into 2 directory 'bacteria' and 'virus'. We also resized and images to 300x300 and center cropped the images with the script 'CropMaxSquare.py'
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
We noticed that a significant x-ray images consist man-made objects, such as medical equipments. To avoid potential data leakage, we mannually deleted. 
![maxCropEffect](docs/screenshots/maxCrop.PNG)



