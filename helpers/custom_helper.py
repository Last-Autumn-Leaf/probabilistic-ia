from collections import defaultdict
import numpy as np
import skimage.color

from skimage.color import rgb2gray
from skimage.feature import canny, blob_doh, blob_dog

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
    image=image.astype('int32')
    result=[]
    for i in range(MODE_SIZE):
        result.append( np.argmax(np.bincount(image[:,:,i].reshape((256*256)))))
    return np.array(result)

def d_pred_count_f (image):
    image=image.astype('int32')
    result=[]
    for i in range(MODE_SIZE):
        result.append(np.max(np.bincount(image[:,:,i].reshape((256*256)))))
    return np.array(result)

# Only works in RGB
fractal_thr=100
sigma_canny = 8
def fractal_dimension(Z, threshold=None, Canny = False, sigma = None): # Z = images
    if threshold ==None:
        threshold=fractal_thr

    if sigma ==None:
        sigma=sigma_canny
    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    Z = Z.astype('float64')

    if Canny:
        Z = skimage.color.rgb2gray(Z)
        Z = canny(Z,sigma = sigma)
    else:
        Z = rgb2gray(Z)

    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])

    # Transform Z into a binary array
    if not Canny:
        Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    if 0 in counts :
        return 0

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return [-coeffs[0]]*3


#Only works inRGB
def number_of_blob(image,max_sigma=30,th=0.1) :

    image=image.astype('uint8')
    image_gray = rgb2gray(image)
    #blobs_doh = blob_doh(image_gray, max_sigma=max_sigma, threshold=th / 10)
    #return [len(blobs_doh)]*3
    blobs_dog = blob_dog(image_gray, max_sigma=max_sigma, threshold=th)
    blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
    return [len(blobs_dog),np.max(blobs_dog[:, 2]),np.mean(blobs_dog[:, 2])]

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

# Special RGB dims
d_fractal = 'fractal dimension'
d_fractal_f = lambda x: fractal_dimension(Z = x,Canny=False)
d_n_blob = 'numbers of blobs'
d_n_blob_f = lambda x : number_of_blob(x)

# Associative variable means they calculate always using the 3 dimensions of a pixel
v_pred_bin = 'most predominant triplet bin'
v_pred_bin_f=lambda x: getHighestFrequencyVector(x)

#d_square_sum,v_pred_bin
all_var_names=[d_mean_bin,d_pred_bin, d_pred_count,d_fractal,d_square_sum,d_pred_count]

var_name2f= {
    d_mean_bin:d_mean_bin_f,
    d_pred_bin:d_pred_bin_f,
    d_square_sum:d_square_sum_f,
    v_pred_bin:v_pred_bin_f,
    d_pred_count:d_pred_count_f,
    d_fractal:d_fractal_f,
    d_n_blob:d_n_blob_f
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


def arrange_train_data(train_data):
    if type(train_data)== list :
        temp=[]
        for arr in train_data :
            temp+=arr.tolist()
        train_data=np.array(temp)
    else:
        train_data = np.array(train_data)
        x, y, z = train_data.shape
        train_data = train_data.reshape(x*y, z)
    return train_data


