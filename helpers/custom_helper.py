from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from skimage import color as skic
from skimage.feature import canny

np.random.seed(0)
N_CLASSES=4
get_n_rand_from_set = lambda sett, n=1 :np.random.choice(sett, n)
def getHighestFrequencyVector(image):
    store = defaultdict(int)
    x, y, z = image.shape
    max_bin = 0
    for i in range(x):
        for j in range(y):
            a=tuple(image[i, j])
            store[a] += 1
            if max_bin <store[a] :
                max_bin=store[a]
                max_vector=a

    return np.append(np.array(max_vector),max_bin)

def d_pred_bin_f (image):
    result=[]
    for i in range(MODE_SIZE):
        result.append( np.argmax(np.bincount(image[:,:,i].reshape((256*256)))))
    return np.array(result)

def d_pred_count_f (image):
    result=[]
    for i in range(MODE_SIZE):
        result.append(np.max(np.bincount(image[:,:,i].reshape((256*256)))))
    return np.array(result)


def fractal_dimension(array, max_box_size=None, min_box_size=1, n_samples=20, n_offsets=0, plot=False):
    """Calculates the fractal dimension of a 3D numpy array.
        Pour nous, on le fait sur 2D array obtenu par edge detection de canny
    Args:
        array (np.ndarray): The array to calculate the fractal dimension of.
        max_box_size (int): The largest box size, given as the power of 2 so that
                            2**max_box_size gives the sidelength of the largest box.
        min_box_size (int): The smallest box size, given as the power of 2 so that
                            2**min_box_size gives the sidelength of the smallest box.
                            Default value 1.
        n_samples (int): number of scales to measure over.
        n_offsets (int): number of offsets to search over to find the smallest set N(s) to
                       cover  all voxels>0.
        plot (bool): set to true to see the analytical plot of a calculation.


    """
    ###

    array = array.astype('float64')
    array = skic.rgb2gray(array)
    array = canny(array)
    # plt.imshow(array)
    # plt.show()
    array = np.expand_dims(array,axis = 2)
    ###

    # determine the scales to measure on


    if max_box_size == None:
        # default max size is the largest power of 2 that fits in the smallest dimension of the array:
        max_box_size = int(np.floor(np.log2(np.min(array.shape))))
    scales = np.floor(np.logspace(max_box_size, min_box_size, num=n_samples, base=2))
    scales = np.unique(scales)  # remove duplicates that could occur as a result of the floor

    # get the locations of all non-zero pixels
    locs = np.where(array > 0)
    voxels = np.array([(x, y, z) for x, y, z in zip(*locs)])

    # count the minimum amount of boxes touched
    Ns = []
    # loop over all scales
    for scale in scales:
        touched = []
        if n_offsets == 0:
            offsets = [0]
        else:
            offsets = np.linspace(0, scale, n_offsets)
        # search over all offsets
        for offset in offsets:
            bin_edges = [np.arange(0, i, scale) for i in array.shape]
            bin_edges = [np.hstack([0 - offset, x + offset]) for x in bin_edges]
            H1, e = np.histogramdd(voxels, bins=bin_edges)
            touched.append(np.sum(H1 > 0))
        Ns.append(touched)
    Ns = np.array(Ns)

    # From all sets N found, keep the smallest one at each scale
    Ns = Ns.min(axis=1)

    # Only keep scales at which Ns changed
    scales = np.array([np.min(scales[Ns == x]) for x in np.unique(Ns)])

    Ns = np.unique(Ns)
    Ns = Ns[Ns > 0]
    scales = scales[:len(Ns)]
    # perform fit
    coeffs = np.polyfit(np.log(1 / scales), np.log(Ns), 1)

    # make plot
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(np.log(1 / scales), np.log(np.unique(Ns)), c="teal", label="Measured ratios")
        ax.set_ylabel("$\log N(\epsilon)$")
        ax.set_xlabel("$\log 1/ \epsilon$")
        fitted_y_vals = np.polyval(coeffs, np.log(1 / scales))
        ax.plot(np.log(1 / scales), fitted_y_vals, "k--", label=f"Fit: {np.round(coeffs[0], 3)}X+{coeffs[1]}")
        ax.legend();
    return (coeffs[0])*3

# ------ MODE NAMES :
RGB = 'RGB'
HSV = 'HSV'
Lab = 'Lab'
MODE_SIZE=3
all_modes=[RGB,HSV,Lab]
class2detailed_repr = {RGB: ['Red', 'Green', 'Blue','f'],
                       HSV: ['Hue', 'Saturation', 'Value','f'],
                       Lab: ['L', 'a', 'b','f']}


# Variables names and # Fonctions
# Dissociative variables means they calculate without taking the other dimension into account
d_mean_bin = 'mean bin'
d_mean_bin_f = lambda x : np.mean(x,axis=(0,1))
d_pred_bin = 'most predominant bin'

d_pred_count = 'counting occurence of predominant bin'


d_square_sum = 'square sum'
d_square_sum_f = lambda x: np.sum(np.square(x)) / 256 ** 3

d_fractal = 'fractal dimension'
d_fractal_f = lambda x: fractal_dimension(array = x)
# Associative variable means they calculate always using the 3 dimensions of a pixel
v_pred_bin = 'most predominant triplet bin'
v_pred_bin_f=lambda x: getHighestFrequencyVector(x)

#d_square_sum,v_pred_bin
all_var_names=[d_mean_bin,d_pred_bin, d_pred_count,d_fractal]

var_name2f= {
    d_mean_bin:d_mean_bin_f,
    d_pred_bin:d_pred_bin_f,
    d_square_sum:d_square_sum_f,
    v_pred_bin:v_pred_bin_f,
    d_pred_count:d_pred_count_f,
    d_fractal:d_fractal_f
}
associative_dims=[v_pred_bin]

from contextlib import contextmanager
import time
from datetime import  timedelta
@contextmanager
def timeThat(name=''):
    try:
        start = time.time()
        yield ...
    finally:
        end = time.time()
        print(name+' finished in ',timedelta(seconds=end-start))


CLASS_COLOR_ARRAY=['blue','green','black','red']
