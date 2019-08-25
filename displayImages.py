import matplotlib.pyplot as plt
import numpy as np

def display2dImages(imageList, title=None, shareAxes=False):
    """ Displays 2D images together """
    fig = plt.figure()

    ax1 = None

    for count, image in enumerate(imageList):
        if ax1 is None:
            ax1 = fig.add_subplot(1,len(imageList),count+1)
            ax1.imshow(image,'gray')
        else:
            if shareAxes:
                ax = fig.add_subplot(1,len(imageList),count+1, sharex=ax1, sharey=ax1)
            else: 
                ax = fig.add_subplot(1,len(imageList),count+1)
            ax.imshow(image,'gray')
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    if title is not None:
        plt.title(title)
    plt.show()

def display3dImages(imageList, mode='None'):
    """ Displays a list of 3D image """
    fig = plt.figure()
    for count, image in enumerate(imageList):
        nRows, nCols = image.shape

        xCoords = np.flip(np.arange(nCols) * -1)
        yCoords = np.arange(nRows)

        xMesh,yMesh = np.meshgrid(xCoords,yCoords)
        
        ax = fig.add_subplot(1,len(imageList),count+1,projection='3d')
        if mode is 'scatter':
            ax.scatter(xMesh, yMesh, image)
        elif mode is 'surface':
            ax.plot_surface(xMesh, yMesh, image, cmap='plasma')

        else:
            print("ERROR: Specify 3D image mode")
        
    plt.show()
