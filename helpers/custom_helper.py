import glob
import os
from collections import defaultdict
import numpy as np
import skimage.color

from skimage.feature import canny, blob_doh, blob_dog
from skimage import io as skiio
import matplotlib.pyplot as plt
import seaborn as sns

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
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

fractal_thr=100
sigma_canny = 8
def fractal_dimension(Z, threshold=None, Canny = False, sigma = None): # Z = images
    if threshold ==None:
        threshold=fractal_thr

    if sigma ==None:
        sigma=sigma_canny

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


def d_n_edges_sum_f(image,threshold=75):
    image = image.astype('uint8')
    image = rgb2gray(image)
    #image =(image < threshold)
    image=canny(image,np.sqrt(2))
    return [np.sum(image)]*3


#Only works inRGB
def number_of_blob(image,max_sigma=30,th=0.1) :

    image=image.astype('uint8')
    image_gray = skimage.color.rgb2gray(image)

    '''blobs_doh = blob_doh(image_gray, max_sigma=max_sigma, threshold=th / 10)
    if len(blobs_doh)!=0 :
        return [len(blobs_doh),np.max(blobs_doh[:, 2]),np.mean(blobs_doh[:, 2])]
    else :
        return [0]*3'''

    # using DOG
    blobs_dog = blob_dog(image_gray, max_sigma=max_sigma, threshold=th)
    blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
    if len(blobs_dog)!=0 :
        return [len(blobs_dog),np.max(blobs_dog[:, 2]),np.mean(blobs_dog[:, 2])]
    else :
        return [0]*3

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


d_n_edges_sum='sum of edges'
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
    d_n_blob:d_n_blob_f,
    d_n_edges_sum:d_n_edges_sum_f
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


CLASS_COLOR_ARRAY=['blue','green','black']+(['red'] if N_CLASSES==4 else [])
class_labels=['coast','forest','street']+(['sunset'] if N_CLASSES==4 else [])


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




PREVENT_OS_SORT = True
def load_images():
    image_folder = r"." + os.sep + "baseDeDonneesImages"
    _path = glob.glob(image_folder + os.sep + r"*.jpg")
    # To not be depedent of the OS-sort
    if PREVENT_OS_SORT:
        _path.sort()
        np.random.shuffle(_path)
    return _path

useStoredBlob = True
DEFAULT_BLOB_BINS=256
stored_blob_path=f'../problematique/blob_{DEFAULT_BLOB_BINS}.npy'
def storeBlobData(path=stored_blob_path,n_bins=DEFAULT_BLOB_BINS):
    arr =np.zeros((3,980))
    images = np.array([np.array(skiio.imread(image)) for image in load_images()])

    for i,image in enumerate(images) :
        image = np.round(image / np.max(image) * (n_bins - 1)).astype('int32')
        arr[0:4,i]=d_n_blob_f(image)
    np.save(path,arr)

def loadStoreBlobData(path=stored_blob_path):

    if not os.path.exists(path) : # Security check
        storeBlobData()
    return np.load(path)

def confusion_matrix(pred,target, n_classes = N_CLASSES):
    # Calcul de la matrice de confusion
    confus_mat = np.zeros((n_classes, n_classes))  # Intialisation de la matrice au nombre de symboles disponbles
    pred, target=pred.astype('uint8'),target.astype('uint8')
    for i in range(len(target)):
        confus_mat[pred[i],target[i]] += 1

    return confus_mat

def plot_confusion_matrix(df_confusion, labels=class_labels, title='Confusion matrix',ax=None):
    if ax ==None:
        ax = plt.subplot()
    sns.heatmap(df_confusion, cmap=plt.get_cmap('Blues'), annot=True, fmt='g',
                ax=ax);  # annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Target labels')
    ax.set_ylabel('Predicted labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

N_KMEAN=6
N_KNN=5
KNN_MODEL_PATH=(f'../model/KNN_MODEL_{N_KMEAN}.npy',f'../model/KNN_MODEL_{N_KNN}.npy') if N_CLASSES==4 else  (f'../model/KNN_3MODEL_{N_KMEAN}.npy',f'../model/KNN_3MODEL_{N_KNN}.npy')