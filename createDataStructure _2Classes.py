import os
import glob
import random
from PIL import Image
from shutil import copyfile,copy

train_test_split = 0.3
random.seed(21)
root_source = 'center_crop_clean_300'
resize_resolution = 200
root_normal_source = root_source+'/normal/'
root_bacteria_source = root_source+'/bacteria/'
root_virus_source = root_source+'/virus/'

root_export = 'data_2C_'+'from_'+root_source+'_to_'+str(resize_resolution)+'/'
if not os.path.exists(root_export):
    os.makedirs(root_export)

train_export = root_export+'train/'
if not os.path.exists(train_export):
    os.makedirs(train_export)

validation_export = root_export+'validation/'
if not os.path.exists(validation_export):
    os.makedirs(validation_export)

train_normal_export = train_export+'normal/'
if not os.path.exists(train_normal_export):
    os.makedirs(train_normal_export)

train_pneumonia_export = train_export+'pneumonia/'
if not os.path.exists(train_pneumonia_export):
    os.makedirs(train_pneumonia_export)


validation_normal_export = validation_export+'normal/'
if not os.path.exists(validation_normal_export):
    os.makedirs(validation_normal_export)

validation_pneumonia_export = validation_export+'pneumonia/'
if not os.path.exists(validation_pneumonia_export):
    os.makedirs(validation_pneumonia_export)



def saveImgs(imgList, imgPath):
    for eachImg in imgList:
        im = Image.open(eachImg)
        #print(im.size)
        # copy(eachImg, imgPath)
        im_rz = im.resize((resize_resolution, resize_resolution), Image.ANTIALIAS)
        #print(imgPath + eachImg.split('\\')[-1])
        im_rz.save(imgPath + eachImg.split('\\')[-1])


def listDiff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

normalImgList = glob.glob(root_normal_source + "*.jpeg")
bacteriaImgList = glob.glob(root_bacteria_source + "*.jpeg")
virusImgList = glob.glob(root_virus_source + "*.jpeg")


normalImgList_train = random.sample(normalImgList, k = int(len(normalImgList)*(1-train_test_split)))
#print(normalImgList_train)
saveImgs(normalImgList_train,train_normal_export)

normalImgList_validation = listDiff(normalImgList,normalImgList_train)
saveImgs(normalImgList_validation,validation_normal_export)


bacteriaImgList_train = random.sample(bacteriaImgList, k = int(len(bacteriaImgList)*(1-train_test_split)))
saveImgs(bacteriaImgList_train,train_pneumonia_export)

bacteriaImgList_validation = listDiff(bacteriaImgList,bacteriaImgList_train)
saveImgs(bacteriaImgList_validation,validation_pneumonia_export)


virusImgList_train = random.sample(virusImgList, k = int(len(virusImgList)*(1-train_test_split)))
saveImgs(virusImgList_train,train_pneumonia_export)

virusImgList_validation = listDiff(virusImgList,virusImgList_train)
saveImgs(virusImgList_validation,validation_pneumonia_export)


print (len(normalImgList),len(normalImgList_train),len(normalImgList_validation))
print(len(bacteriaImgList),len(bacteriaImgList_train),len(bacteriaImgList_validation))
print(len(virusImgList),len(virusImgList_train),len(virusImgList_validation))