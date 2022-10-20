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


# ------ MODE NAMES :
RGB = 'RGB'
HSV = 'HSV'
Lab = 'Lab'
class2detailed_repr = {RGB: ['Red', 'Green', 'Blue','f'],
                       HSV: ['Hue', 'Saturation', 'Value','f'],
                       Lab: ['L', 'a', 'b','f']}


# Variables names and # Fonctions
# Dissociative variables means they calculate without taking the other dimension into account
d_mean_bin = 'mean bin'
d_mean_bin_f = lambda x : np.mean(x,axis=(0,1))
d_pred_bin = 'most predominant bin'

d_square_sum = 'square sum'
d_square_sum_f = lambda x: np.sum(np.square(x)) / 256 ** 3

# Associative variable means they calculate always using the 3 dimensions of a pixel
v_pred_bin = 'most predominant triplet bin'
v_pred_bin_f=lambda x: getHighestFrequencyVector(x)

all_var_names=[d_mean_bin,d_pred_bin,d_square_sum,v_pred_bin]

var_name2f= {
    d_mean_bin:d_mean_bin_f,
    d_pred_bin:d_pred_bin_f,
    d_square_sum:d_square_sum_f,
    v_pred_bin:v_pred_bin_f
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
        return f"{self.name},{'associative' if self.isAssociative else'dissociative'}," \
               f"{'avg' if self.isAvg else'no-avg'},{self.max_dataset_size}"
# Default values of dims
var_name2default_dim= {
    d_mean_bin:dimension(d_mean_bin,True),
    d_pred_bin:dimension(d_pred_bin,True),
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

    def compute_for_image(self,image,index):
        for var in self.variables :
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

class ClassesTracker :
    def __init__(self,name,idx_list):
        self.name = name
        self.idx_list=idx_list

    def __len__(self):
        return len(self.idx_list)


