import glob
import os
from collections import defaultdict
import numpy as np


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
    for i in range(3):
        result.append( np.argmax(np.bincount(image[:,:,i].reshape((256*256)))))
    return np.array(result)

def d_pred_count_f (image):
    result=[]
    for i in range(3):
        result.append(np.max(np.bincount(image[:,:,i].reshape((256*256)))))
    return np.array(result)


# ------ MODE NAMES :
RGB = 'RGB'
HSV = 'HSV'
Lab = 'Lab'
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

# A dimension is class-wise
# 4*n if isAssociative else 3*n
class dimension:
    # using Numpy will optimize the space and reduce the computation time
    def __init__(self, name, isAvg=True, max_dataset_size=292,mode=RGB):
        self.max_dataset_size = max_dataset_size
        self.name = name
        self.mode=mode
        self.isAssociative = self.name in associative_dims
        # Doing the average only when asked to save computation time
        self.isAvg = isAvg
        self.round_val = 1
        self.create_storage()

    def change_mode(self,mode):
        self.mode=mode
    def create_storage(self):
        self.data = np.zeros((4 if self.isAssociative else 3, self.max_dataset_size))
        if self.isAvg:
            self.mean = np.zeros((self.data.shape[0], 1))

    def update_dataset_size(self,new_size):
        self.max_dataset_size = new_size
        self.create_storage()
        return self

    def compute_mean(self):
        assert self.isAvg, "Not a average variable !"
        self.mean = np.round(np.mean(self.data, axis=1), self.round_val)

    def __setitem__(self, key, value):
        self.data[:,key]=value

    def __getitem__(self, item):
        return self.data[:,item]

    def get_axis(self,axe,mean=False):
        if not mean :
            return self.data[axe]
        elif self.isAvg:

            return self.mean[axe]


    def get_mean(self,index):
        assert self.isAvg,'Not an average variable'
        return self.mean[index]

    def __repr__(self):
        return f"{self.name},{self.mode},{'associative' if self.isAssociative else'dissociative'}," \
               f"{'avg' if self.isAvg else'no-avg'},{self.max_dataset_size}"
# Default values of dims
var_name2default_dim= {
    d_mean_bin:dimension(d_mean_bin,True),
    d_pred_bin:dimension(d_pred_bin,True),
    d_pred_count:dimension(d_pred_count,False),
    d_square_sum:dimension(d_square_sum,True),
    v_pred_bin:dimension(v_pred_bin,False)
}
getDefaultVar = lambda name,dataset_size=2 : var_name2default_dim[name].update_dataset_size(dataset_size)

class VariablesTracker:

    def __init__(self,track_list):
        if type(track_list)!= list :
            track_list=[track_list]
        self.variables=track_list # list of dimensions to watch !
        self.name2var={var.name:var for var in self.variables}


    def __getitem__(self, item): # get a variable by index
        return self.variables[item]

    def getVarByName(self,name):
        return self.name2var[name]

    def update_dataset_size(self,new_size):
        self.variables=[var.update_dataset_size(new_size) for var in self.variables]
        self.name2var={var.name:var for var in self.variables}
    def compute_mean(self):
        for var in self.variables:
            if var.isAvg :
                var.compute_mean()

    def compute_for_image(self,image,index,var):
            # We should only attack one var at a time
            debug=var_name2f[var.name](image)
            var[index]=debug

    def pick_var(self,dim=d_mean_bin, mode='RGB', index_mode=0):
        for var in self.variables :
            if var.name==dim  and var.mode==mode:
                return var.get_axis (index_mode)

    def pick_var_mean(self,dim=d_mean_bin, mode='RGB', index_mode=0):
        for var in self.variables :
            if var.name==dim  and var.mode==mode:
                return var.get_axis (index_mode,True )


    def __repr__(self):
        s=""
        for var in self.variables :
            s+=f"{var}" +"\n"
        return s


from skimage import io as skiio
from skimage import color as skic
from sklearn.model_selection import train_test_split as ttsplit
import helpers.analysis as an
class ClassesTracker :
    def __init__(self):
        # liste de toutes les images
        image_folder = r"." + os.sep + "baseDeDonneesImages"
        _path = glob.glob(image_folder + os.sep + r"*.jpg")
        image_list = os.listdir(image_folder)
        # Filtrer pour juste garder les images
        image_list = [i for i in image_list if '.jpg' in i]
        self.images = np.array([np.array(skiio.imread(image)) for image in _path])
        self.n_class=3
        self.coast_id=[]
        self.forest_id=[]
        self.street_id=[]
        self.class_labels=np.zeros((len(image_list),1)).astype('int32')
        for i, name_file in enumerate(image_list):
            if "coast" in name_file:
                self.coast_id.append(i)
                self.class_labels[i]=0
            elif "forest" in name_file:
                self.forest_id.append(i)
                self.class_labels[i]=1
            else:
                self.street_id.append(i)
                self.class_labels[i]=2

        self.all_classes= [self.coast_id,self.forest_id,self.street_id]


        temp = ttsplit( [i for i in range (len(self.class_labels))],
                        self.class_labels, test_size=0.2, shuffle=True)
        self.training_data_idx = temp[0]
        self.training_target = temp[2]
        self.validation_data_idx = temp[1]
        self.validation_target = temp[3]

        #This should be coherent
        dimensions_list = [dimension(name = d_pred_bin,mode = HSV),dimension(name = d_pred_count,mode = HSV)]
        self.dims_list=[(d_pred_bin,HSV,0),(d_pred_count,HSV,0)]

        self.tracker = VariablesTracker(dimensions_list)
        self.tracker.update_dataset_size(len(self.images))

        self.n_bins=256
        with timeThat('Pre processing of all the data'):
            self.pre_process_all_data()

        plist =[self.tracker.pick_var(dim[0],dim[1],dim[2]) for dim in self.dims_list]
        self.extent=an.Extent(ptList=np.stack(plist,axis=1))

        # génération de données aléatoires
        ndonnees = 5000
        self.donneesTest = an.genDonneesTest(ndonnees, self.extent)


    def pre_process_all_data(self):
        def rescaleHistLab( LabImage, n_bins):
            """
            Helper function
            La représentation Lab requiert un rescaling avant d'histogrammer parce que ce sont des floats!
            """

            # Constantes de la représentation Lab
            class LabCte:  # TODO JB : utiliser an.Extent?
                min_L: int = 0
                max_L: int = 100
                min_ab: int = -110
                max_ab: int = 110

            # Création d'une image vide
            imageLabRescale = np.zeros(LabImage.shape)
            # Quantification de L en n_bins niveaux
            imageLabRescale[:, :, 0] = np.round(
                (LabImage[:, :, 0] - LabCte.min_L) * (n_bins - 1) / (
                        LabCte.max_L - LabCte.min_L))  # L has all values between 0 and 100
            # Quantification de a et b en n_bins niveaux
            imageLabRescale[:, :, 1:3] = np.round(
                (LabImage[:, :, 1:3] - LabCte.min_ab) * (n_bins - 1) / (
                        LabCte.max_ab - LabCte.min_ab))  # a and b have all values between -110 and 110
            return imageLabRescale


        should_Compute = [False, False, False]
        for var in self.tracker.variables:
            if var.mode == RGB:
                should_Compute[0] = True
            elif var.mode == HSV:
                should_Compute[1] = True
            elif var.mode == Lab:
                should_Compute[2] = True

        for i,images in enumerate(self.images):

            if should_Compute[1]:
                images_HSV = skic.rgb2hsv(images)
                images_HSV = np.round(images_HSV / np.max(images_HSV) * (self.n_bins - 1)).astype('int32')
                for var in self.tracker :
                    if var.mode==HSV :
                        self.tracker.compute_for_image(images_HSV, i,var)

            if should_Compute[2]:
                # L [0,100],a,b [-127,127]
                images_LAB = skic.rgb2lab(images)
                images_LAB = rescaleHistLab(images_LAB, self.n_bins)
                images_LAB = images_LAB.astype('int32')
                for var in self.tracker :
                    if var.mode==Lab :
                        self.tracker.compute_for_image(images_LAB, i,var)
            if  should_Compute[0]:
                images = np.round(images / np.max(images) * (self.n_bins - 1)).astype('int32')
                for var in self.tracker :
                    if var.mode==RGB :
                        self.tracker.compute_for_image(images, i,var)



        self.tracker.compute_mean()

        return self.tracker

    def get_data(self,train=True):
        if train :
            idx=self.training_data_idx
        else:
            idx=self.validation_data_idx

        data=np.zeros( (len(idx),len(self.dims_list)) )
        for i,dim in enumerate(self.dims_list) :
            var = dim[0]
            mode = dim[1]
            index = dim[2]
            data[:,i]=self.tracker.pick_var(var,mode,index)[idx]
        return data
    def get_training_data(self):
        return self.get_data()

    def get_test_data(self):
        return self.get_data(False)

    def get_data_classwise(self,n=250):

        classes=[ self.all_classes[i][:n] for i in range(3)]
        target=np.array([j for j in range(3) for i in range(n)])[:,None]

        val_classes=[ self.all_classes[i][n:] for i in range(3)]

        val_target=[]
        val_idx=[]
        for i,val in enumerate(val_classes):
            val_idx+=val
            val_target+= [i]*len(val)
        val_target=np.array(val_target)[:,None]
        val_data=np.zeros((len(val_idx),len(self.dims_list)))

        data = np.zeros((self.n_class, n,len(self.dims_list)))

        for i, dim in enumerate(self.dims_list):
            var = dim[0]
            mode = dim[1]
            index = dim[2]

            temp=self.tracker.pick_var(var, mode, index)
            val_data[:,i]=temp[val_idx]
            for j,class_idx in enumerate(classes):
                data[j,:,i] = temp[class_idx]

        return data,target,val_data,val_target

    def get_target_data(self,train=True):
        return self.class_labels[self.training_data_idx
            if train else self.validation_data_idx]


    def __len__(self):
        return len(self.images)

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
