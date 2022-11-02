from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters._sparse import correlate_sparse
from skimage.io import imshow, show

from helpers.custom_helper import load_images, sigma_canny
from skimage import io as skiio
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
        images = np.array([np.array(skiio.imread(image)) for image in load_images()])
        image=images[0]
        image=image.astype('uint8')
        image=rgb2gray(image)
        image=canny(image,sigma_canny )
        np.sum(image)
        #return [np.sum(image)]*3