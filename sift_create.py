import cv2
import pickle
import os
from time import time
import numpy as np

dim = (244,244)

siftDict = {}

orb = cv2.ORB_create()



def calculateHOGpointsSingleImg(orb,img):
  
    assert (img is not None)
    
    kpts, descs = orb.detectAndCompute(img,None)

    return (kpts,descs)


def imgArtCropper(img):
    
    if type(img) is np.ndarray:
        width,height = img.shape
        img = img[int(0.2*height):int(0.7*height),int(0.2*width):int(0.8*width)]
    else:
        width, height = img.size
        img = img.crop((int(0.2*width), int(0.2*height), int(0.8*width), int(0.7*height))) 
        
    return img




i = 0
totalTime = time()
for key,value in featureMapDict.items():   
    print('--------- ', i, ' ---------')
    begin = time()
    
    absPath,output2 = value
    #img = cv2.imread('.'+absPath)
    img = imgArtCropper(cv2.imread('.'+absPath,0))
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    _,descs = calculateHOGpointsSingleImg(orb,img)
    siftDict[key] = (descs, absPath) 
    print(time()-begin,'s')
    i += 1
print('Total time: ',time()-totalTime,'s')
savePath = './siftMap-072320.pkl'
f = open(savePath,"wb")
pickle.dump(siftDict,f)
f.close()

loadDict = pickle.load(open(savePath, 'rb'))
print(len(loadDict))