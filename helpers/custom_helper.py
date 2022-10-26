from collections import defaultdict
import numpy as np
np.random.seed(0)
N_CLASSES=3
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

# Associative variable means they calculate always using the 3 dimensions of a pixel
v_pred_bin = 'most predominant triplet bin'
v_pred_bin_f=lambda x: getHighestFrequencyVector(x)

#d_square_sum,v_pred_bin
all_var_names=[d_mean_bin,d_pred_bin, d_pred_count]

var_name2f= {
    d_mean_bin:d_mean_bin_f,
    d_pred_bin:d_pred_bin_f,
    d_square_sum:d_square_sum_f,
    v_pred_bin:v_pred_bin_f,
    d_pred_count:d_pred_count_f
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


CLASS_COLOR_ARRAY=['blue','green','black']
