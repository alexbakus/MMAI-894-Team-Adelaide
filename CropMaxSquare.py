import glob
import os
from PIL import Image
from operator import itemgetter

path_normal = 'NORMAL'
path_pneumnia = 'PNEUMONIA'

# the code trys to corp/export center of the original img so that lung is the focus, but if the picture is too large
# the code will first shrink it down to 1024x1024 with anti-alising filter then corp to limit x limit

limit = 400
zoom_factor = 0.95

root_export = 'center_crop_'+str(limit)+'\\'
if not os.path.exists(root_export):
    os.makedirs(root_export)

normal_export = root_export+'\\normal\\'
if not os.path.exists(normal_export):
    os.makedirs(normal_export)

bacteria_export = root_export+'\\bacteria\\'
if not os.path.exists(bacteria_export):
    os.makedirs(bacteria_export)

virus_export = root_export+'\\virus\\'
if not os.path.exists(virus_export):
    os.makedirs(virus_export)


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img,zoom):
    return crop_center(pil_img, int((min(pil_img.size))*zoom), int((min(pil_img.size))*zoom))


normalImgList = glob.glob(path_normal + "/*.jpeg")
normalExport_count = 0
for eachImg in normalImgList:
    #print (eachNormalImg.split('\\')[1])
    im = Image.open(eachImg)
    width, height = im.size
    if(width >= limit and height >= limit):
        im_center_crop = crop_max_square(im,zoom_factor)
        im_center_crop.save(normal_export+eachImg.split('\\')[1])
        normalExport_count += 1
        #imgSizeList.append((width, height))


virusImgList = glob.glob(path_pneumnia + "/*vir*.jpeg")
virusExport_count = 0
for eachImg in virusImgList:
    #print (eachNormalImg.split('\\')[1])
    im = Image.open(eachImg)
    width, height = im.size
    if(width >= limit and height >= limit):
        im_center_crop = crop_max_square(im,zoom_factor)
        im_center_crop.save(virus_export+eachImg.split('\\')[1])
        #imgSizeList.append((width, height))
        virusExport_count += 1

bacteriaImgList = glob.glob(path_pneumnia + "/*bact*.jpeg")
bacteriaExport_count = 0
for eachImg in bacteriaImgList:
    #print (eachNormalImg.split('\\')[1])
    im = Image.open(eachImg)
    width, height = im.size
    if(width >= limit and height >= limit):
        im_center_crop = crop_max_square(im,zoom_factor)
        im_center_crop.save(bacteria_export+eachImg.split('\\')[1])
        bacteriaExport_count += 1
        #imgSizeList.append((width, height))

print('Normal Img : ',normalExport_count,
      'Bacteria Img : ',bacteriaExport_count,
      'virus Img : ', virusExport_count)


'''print(imgSizeList)
print(len(imgSizeList))
print(min(imgSizeList, key=itemgetter(1))[0])
print(min(imgSizeList, key=itemgetter(1))[1])'''
