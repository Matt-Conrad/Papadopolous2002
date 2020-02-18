import numpy as np
from scipy.ndimage.measurements import label
import cv2
import visualization.display_images as dI

def getBiggestComp(image):
    """ Uses connected components to get the breast """
    structure = np.ones([3,3], dtype=np.int) # Relational matrix (8-connected)
    # Run connected components to label the various connected components
    labeled_image, n_components = label(image, structure=structure) 

    # Loop through the components and get the biggest component
    nPixelsInBiggestComp = 0
    biggestComp = 0
    for i in range(1,n_components+1): # Start at 1 to avoid considering background 
        component = (labeled_image == i)
        pixelsInComp = np.sum(component)
        if pixelsInComp > nPixelsInBiggestComp:
            nPixelsInBiggestComp = pixelsInComp
            biggestComp = component

    # Create binary mask in the shape of the biggest component
    img = np.zeros(image.shape)
    img[biggestComp] = 1
    return img

def ROI(image):
    """ Uses connected components to get possible microcalcification objects """

    """ In the segmented image obtained in the previous stage, neighbouring pixels with
        connectivity of eight are grouped together to create possible microcalcification objects. """
    structure = np.ones([3,3], dtype=np.uint8) # Relational matrix (8-connected)
    # Run connected components to label the various connected components
    labeled_image, n_components = label(image, structure=structure) 

    """ Objects containing one or two pixels are rejected since they are considered as artifacts """
    img = np.copy(image)
    # Loop through the components and remove components <= 2px
    for i in range(1,n_components+1): # Start at 1 to avoid considering background 
        component = (labeled_image == i)
        pixelsInComp = np.sum(component)
        if pixelsInComp <= 2:
            img[component] = 0
    img = img.astype(np.uint8)
    dI.display2dImages([image,img],shareAxes=True,title='Remove components smaller than 3 pixels')

    """ The application of the erosion operator (with structure element a 3x3 kernel of unit value) results in the removal of all
        objects apart from those that have at least one innermost pixel that is not part of their boundary."""
    structure = np.ones([3,3], dtype=np.uint8) # Relational matrix (8-connected)
    erode = cv2.erode(img,structure,iterations=1)
    dI.display2dImages([img,erode],shareAxes=True,title='First erosion')

    """ Only inner pixels that belong to large objects remain. These pixels correspond to the centres of ROIs, which are generated using the dilation operator with a
        3x3 structure element of unit value. The dilation is repeated 50 times in order to produce a ROI with sufficient area around the object."""
    roiImage = cv2.dilate(erode.astype(np.uint8),structure,iterations=50)
    dI.display2dImages([erode,roiImage],shareAxes=True,title='ROIs')

    """ A ROI that is not of minimum size is considered as having been generated from a group of objects located in the same neighbourhood. In such case,
        two or more ROIs will be combined and a new enlarged ROI will be generated containing more than two of the original objects. """
    labeled_image, n_components = label(roiImage, structure=structure)  

    """ The set of ROIs is partitioned in two groups depending on their area. The first group contains those ROIs with areas 
        lower than 20,000 pixels (2x100x100), which is a reliable threshold value discriminating ROIs that are generated from individual objects.
        The second group contains the remaining ROIs which contain at least two nearby objects. This discrimination of ROIs defines a novel 
        feature that will be used at the classification stage."""
    group1 = []
    group2 = []
    for i in range(1,n_components+1): # Start at 1 to avoid considering background 
        component = (labeled_image == i)
        pixelsInComp = np.sum(component)
        if pixelsInComp < 20000:
            group1.append(component)
        else:
            group2.append(component)
    
    ROIs = group1 + group2
    dI.display2dImages(ROIs, title='Individual ROIs')

    """ The existence of an individual object close to a ROI might be a problem in some cases. To resolve it a second dilation process is applied 
        on the previous image, but only to the set of larger ROIs, using a 3x3 structure element in a 50-cycles repeated procedure."""
    newROIs = []
    for roi in group1:
        newROIs.append(roi)

    for roi in group2:
        roiDilate = cv2.dilate(roi.astype(np.uint8),structure,iterations=50)
        roiDilate = roiDilate.astype(np.bool_)
        newROIs.append(roiDilate)

    dI.display2dImages(newROIs, shareAxes=True,title='New ROIs')

    return len(group1) + len(group2) #newROIs.astype(np.bool_)