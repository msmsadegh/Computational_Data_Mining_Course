import os, numpy, PIL
from PIL import Image
import numpy as np
from numpy import newaxis
from time import sleep
import warnings
import pandas as pd
import re
warnings.filterwarnings("ignore")

def calculateDistance(i1, i2):
    return numpy.sum((i1-i2)**2)

PATH = '/home/maghareh/Desktop/ThirdSemester/ComputationalDataMining/HW2'
averageImage = []
averageArr = []
# Access all PNG files in directory
for digit in range(0,10):
    allfiles=os.listdir(PATH+f'/TinyMNIST/train/{digit}/')
    imlist=[PATH+f'/TinyMNIST/train/{digit}/'+filename for filename in allfiles if  filename[-4:] in [".jpg",".JPG"]]

    # Assuming all images are the same size, get dimensions of first image
    w,h=Image.open(imlist[0]).size
    N=len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr=numpy.zeros((h,w),numpy.float)

    # Build up average pixel intensities, casting each image as an array of floats
    for im in imlist:
        imarr=numpy.array(Image.open(im),dtype=numpy.float)
        #imarr = imarr[:,:,newaxis]
        arr=arr+imarr/N

    # Round values in array and cast as 8-bit integer
    arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

    # Generate, save and preview final image
    averageImage.append(Image.fromarray(arr))
    averageArr.append(arr)
    averageImage[digit].save(f"Average{digit}.png")


print(averageArr[4].shape)
# for digit in range(0,10):
#     averageArr.append(numpy.array(Image.open(f'Average{digit}.png'), dtype=numpy.float))

allfiles = os.listdir(PATH+f'/TinyMNIST/test/test/')
testlist = [PATH+f'/TinyMNIST/test/test/' + filename for
           filename in allfiles if filename[-4:] in [".jpg", ".JPG"]]
# print(testlist)
#
#
testDF = pd.read_csv(PATH+'/TinyMNIST/test labels.csv')
print(testDF.columns)
print('len:',len(testDF))
correctValues = 0
for testImage in testlist:

    imageId = re.sub("[^0-9]", "", testImage.split('/')[-1])
    npTest = numpy.array(Image.open(testImage), dtype=numpy.float)
    dist=[]
    for digit in range(0,10):

        dist.append(np.linalg.norm(npTest - averageArr[digit]))
    print('Image ID:',imageId)
    print(dist)
    predictValue =(dist.index(min(dist)))
    print("Predict Value:",predictValue)
    print('RealValue:', int(testDF.loc[testDF['id']==int(imageId)]['category'].values))


    if int(predictValue)==int(testDF.loc[testDF['id']==int(imageId)]['category'].values):
        correctValues+=1
        print(True)
    else:
        print(False)
    print('---------------------------')

print(correctValues)
print(len(testDF))
print("Percent:",(correctValues/len(testDF))*100)
#
#     print(npTest.shape)
#
#     breakpoint()
# #print(averageArr[0].shape)

