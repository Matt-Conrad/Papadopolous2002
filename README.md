# Implementation of CAD Paper by Papadopolous (2002)

This project is an implementation of the paper "An automatic microcalcification detection system based on a hybrid neural network classifier" by Papadopolous (2002). The paper can be found here: https://www.sciencedirect.com/science/article/pii/S0933365702000131. 

This code currently only implements sections up to and including 2.3, so the following parts of the algorithm are implemented: 
-preprocessing and segmentation module
-region of interest specification

While I will not include the paper, for each command in the script I've added the relevant sentences in the comments above each command. 

## Dataset

In the paper, the authors use the MIAS dataset whose website is here: http://peipa.essex.ac.uk/info/mias.html. This paper is only interested in detecting calcifications, so we're only looking at the 25 images in the dataset that have calcifications (see calc-mias folder for those images). Also, the CALC.csv show the metadata for each image and the corresponding location of the 30 calcifications in this image subset. 

Interestingly, the paper says the MIAS dataset has 20 images with 25 clusters, which contradicts what I found in the current dataset. This may be due to the fact this article is from 17 years ago.

## Python Modules required

- numpy (1.16.4)
- matplotlib.pyplot (3.1.0)
- scipy (1.3.0)
- opencv (3.4.2)

## Notes

- Currently, the output of the ROI specification seems to have very large ROIs and doesn't seem to work well enough to move onto the classification phase
- The first step in the algorithm, adding a black border, was added because there always seems to be a white line between the breast area and other large objects (like markers) in the image. This white line caused issues when I binary threshold in the 2nd step of the algorithm.

## Acknowledgments

Papadopoulos, A., Fotiadis, D. and Likas, A. (2002). An automatic microcalcification detection system based on a hybrid neural network classifier. Artificial Intelligence in Medicine, 25(2), pp.149-167.
