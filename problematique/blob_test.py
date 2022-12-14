import glob
import os
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage import io as skiio

import matplotlib.pyplot as plt

from helpers.custom_helper import timeThat, storeBlobData


def main(path,max_sigma=30,th=0.1):
    #image = data.hubble_deep_field()[0:500, 0:500]
    image = skiio.imread(path)
    image_gray = rgb2gray(image)

    blobs_log = blob_log(image_gray, max_sigma=max_sigma, num_sigma=10, threshold=th)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image_gray, max_sigma=max_sigma, threshold=th)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image_gray, max_sigma=max_sigma, threshold=th/10)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()


def number_of_blob(image,max_sigma=30,th=0.1) :
    image_gray = rgb2gray(image)
    blobs_doh = blob_doh(image_gray, max_sigma=max_sigma, threshold=th / 10)
    return len(blobs_doh)


def time_numblob_fun():
    image_folder = r"." + os.sep + "baseDeDonneesImages"
    _path = glob.glob(image_folder + os.sep + r"*.jpg")
    with timeThat():
        for path in _path:
            image = skiio.imread(path)
            number_of_blob(image)

if __name__ == '__main__':
    with timeThat('storing blob in'):
        storeBlobData()