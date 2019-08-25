import numpy as np

def blackBorder(image,thickness):
    """ Builds a black border into the edges of the given image """
    bordered = np.copy(image)

    bordered[:thickness,:] = 0 # Top
    bordered[:,:thickness] = 0 # Left
    bordered[-1:, :] = 0 # Bottom
    bordered[:, -1:] = 0 # Right

    return bordered