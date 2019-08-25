import pgmReader as pgm
import os
import displayImages as dI
import morphological as morph
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import blackBorder as bb
import scipy.ndimage.filters as filters
import random

def preprocessImage(image):
    """ Preprocesses an image """
    
    """ add black border to combat white line along some borders """
    bordered = bb.blackBorder(image,1)  # NOTE: NOT A PART OF THE ALGORITHM
    dI.display2dImages([image,bordered],title='Black Border')

    """ Set equal to zero the image pixels with intensity less than 20 """
    binImage = (bordered >= 20)
    dI.display2dImages([image,binImage],title='Zero intensities less than 20')

    """ Neighbouring white pixels with connectivity of eight are grouped together to form objects corresponding
        either to the breast region or to marks and film artifacts. The largest object corresponds to
        the breast region """
    bigComp = morph.getBiggestComp(binImage).astype(np.uint8) 
    dI.display2dImages([binImage,bigComp],title='Biggest Component')

    """ Apply morphological dilation with a structure element radius of 30 pixels (1.5 mm) """
    structElem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(60,60)) # TODO: Not a perfect circle
    dilation = cv2.dilate(bigComp,structElem,iterations=1)
    dI.display2dImages([bigComp,dilation],title='Dilate Breast profile')

    """ All the pixels that do not belong to the expanded breast area are set to zero, resulting in the removal of background,
        marks and artifacts. """
    noBackground = np.multiply(bordered,dilation)
    dI.display2dImages([bordered,noBackground],title='Zero everything not in breast profile')
    # dI.display3dImages([image],'surface')

    """ The minimum rectangle containing the breast region is automatically drawn and it is used in the subsequent processing stages. """
    boundingBoxImage, x, y, boxWidth, boxHeight = boundingBox(noBackground)
    dI.display2dImages([noBackground, boundingBoxImage],title='Bounding box')

    """ EXPERIMENTAL: CROP THE BREAST OUT """
    croppedImage = noBackground[y:y+boxHeight,x:x+boxWidth]
    dI.display2dImages([noBackground, croppedImage],title='Crop out breast')

    """ The whole image is split into 30x30 sub-regions and, using bicubic interpolation, a second plot is obtained representing the
        intensity level of the local background (Fig. 3b). """
    interpImage = interpolate2(croppedImage)
    dI.display2dImages([croppedImage, interpImage],title='Bicubic interpolation')
    # dI.display3dImages([interpImage],'surface')

    """ The interpolated image is subtracted from the original mammogram producing a third image with each pixel value providing the
        difference between the original and local background pixel values. """
    thirdImage = croppedImage.astype(np.int32) - interpImage.astype(np.int32) 
    dI.display2dImages([noBackground, thirdImage],title='Difference image')

    """ The pixels with positive values are identified and a percentage of them (5%) with the highest values is selected
        producing a binary image and also specifying a threshold value (the lowest value among the selected pixels). If 
        the amount of the selected pixels is lower than 10% of the total number of pixels of the cropped mammogram, the 
        pixels with intensity higher than half of the previously specified threshold are added. """
    refinedThirdImage = topIntensityImage(thirdImage) # Binary image (A)
    dI.display2dImages([croppedImage, refinedThirdImage], shareAxes=True,title='Image A')
    
    """ Next a contrast enhancement filter is applied with 9x9 kernel having central element
        equal to 80 and all the other elements equal to 1 """
    kernel = np.ones((9,9),dtype=np.int32) * -1
    kernel[4,4] = 80
    contrastEnhanced = filters.convolve(croppedImage.astype(np.float64),kernel)
    dI.display2dImages([croppedImage, contrastEnhanced],shareAxes=True,title='Contrast enhancement filter')

    """ Five percent of the pixels having the highest intensity are selected, producing a second binary image (B). """
    percentile = 95
    threshold = np.percentile(contrastEnhanced, percentile) 
    topContrastEnhanced = (contrastEnhanced >= threshold).astype(np.uint8) # Binary image (B) 
    dI.display2dImages([croppedImage, topContrastEnhanced],shareAxes=True,title='Image B')

    """ The outcome of the segmentation module is an image produced by the logical summation (AND) of the two
        binary images A and B. """
    A = refinedThirdImage.astype(np.bool_)
    B = topContrastEnhanced.astype(np.bool_)
    C = np.logical_and(A,B)
    dI.display2dImages([A, B, C], shareAxes=True, title='Image C')

    ### ROI SPECIFICATION
    result = morph.ROI(C)
    return result

def interpolate2(image):
    rows, cols = image.shape
    lastRow = rows - 1
    lastCol = cols - 1

    # Create sample grid coordinates
    xStep = 29
    yStep = 29
    xCoords1D = np.arange(0,image.shape[1],xStep,dtype=np.uint32)
    yCoords1D = np.arange(0,image.shape[0],yStep,dtype=np.uint32)
    xCoords1D = np.append(xCoords1D,image.shape[1]-1).astype(np.uint32) # End patches are smaller in order to preserve patch size of 30x30
    yCoords1D = np.append(yCoords1D,image.shape[0]-1).astype(np.uint32)

    # 2D sample grid coordinates
    xCoords2D, yCoords2D = np.meshgrid(xCoords1D,yCoords1D)
    zCoords2D = image[yCoords2D,xCoords2D] # Samples at each patch corner

    # 2D interpolation point coordinates
    xCoords1D2 = np.arange(0,lastCol+1,dtype=np.uint32)
    yCoords1D2 = np.arange(0,lastRow+1,dtype=np.uint32)
    xCoords2D2, yCoords2D2 = np.meshgrid(xCoords1D2,yCoords1D2)
    
    # Flatten sample coordinates to input in griddata
    xCoords2DFlat = np.ravel(xCoords2D)
    yCoords2DFlat = np.ravel(yCoords2D)
    zCoords2DFlat = np.ravel(zCoords2D)

    # Apply interpolation
    interpVal = interp.griddata((xCoords2DFlat,yCoords2DFlat),zCoords2DFlat,(xCoords2D2, yCoords2D2), method='cubic')

    # Clip out of Range values
    intensityFloor = np.iinfo(np.uint8).min
    intensityCeiling = np.iinfo(np.uint8).max
    interpVal = np.clip(interpVal, intensityFloor, intensityCeiling).astype(np.uint8)

    # Display to verify
    # dI.display2dImages([image,interpVal])
    # dI.display3dImages([interpVal], mode='surface')

    return interpVal

def topIntensityImage(thirdImage):
    """ Takes the third image and returns an image with only it's top pixels """

    """ The pixels with positive values are identified """
    positivePixelValues = getOnlyPositives(thirdImage)

    """ A percentage of them (5%) with the highest values is selected producing a binary image and also specifying a threshold value """
    percentile = 95
    threshold = np.percentile(positivePixelValues, percentile) 
    topPixelsBinary = (thirdImage >= threshold).astype(np.uint8)

    """ If the amount of the selected pixels is lower than 10% of the total number of pixels of the cropped mammogram, the pixels with intensity higher than
        half of the previously specified threshold are added."""
    nSelectedPixels = np.sum(topPixelsBinary)
    nTotalPixels = thirdImage.size

    if nSelectedPixels < (nTotalPixels * 0.1):
        halfThreshold = threshold / 2
        topPixelsBinary = (thirdImage >= halfThreshold).astype(np.uint8)

    return topPixelsBinary

def getOnlyPositives(image):
    """ Returns the image whose pixel values are only positive """
    imagePositiveMask = (image > 0) 
    positiveImage = image[imagePositiveMask]
    return positiveImage

def boundingBox(image):
    """ Finds the bounding box of the breast region """
    x,y,w,h = cv2.boundingRect(image)
    blank = np.zeros(image.shape)
    cv2.rectangle(blank,(x,y),(x+w,y+h),1,2)
    return blank, x, y, w, h 

dataDirectory = './PapaPaper/calc-mias'
roiCount = 0
filesList = os.listdir(dataDirectory)
random.shuffle(filesList)
for fileName in filesList:
    if fileName.endswith(".pgm"):
        image = pgm.read_pgm(dataDirectory + '/' + fileName, byteorder='<')

        roiCount += preprocessImage(image)
        print(str(roiCount))
