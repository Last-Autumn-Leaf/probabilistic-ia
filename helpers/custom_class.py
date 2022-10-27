import numpy as np

from helpers.custom_helper import *

#---------------------- Dimension tracker class
# Default values of dims
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
        self.data = np.zeros((MODE_SIZE+1 if self.isAssociative else MODE_SIZE, self.max_dataset_size))
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

var_name2default_dim= {
    d_mean_bin:dimension(d_mean_bin,True),
    d_pred_bin:dimension(d_pred_bin,True),
    d_pred_count:dimension(d_pred_count,False),
    d_square_sum:dimension(d_square_sum,True),
    v_pred_bin:dimension(v_pred_bin,False)
}
getDefaultVar = lambda name,dataset_size=2 : var_name2default_dim[name].update_dataset_size(dataset_size)


#---------------------- Variables tracker class
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


#---------------------- Classes tracker class
import os
import glob
from skimage import io as skiio
from skimage import color as skic
from sklearn.model_selection import train_test_split as ttsplit
from helpers.analysis import Extent,genDonneesTest
PREVENT_OS_SORT = True
class ClassesTracker :
    def __init__(self):
        # liste de toutes les images
        image_folder = r"." + os.sep + "baseDeDonneesImages"
        _path = glob.glob(image_folder + os.sep + r"*.jpg")
        # To not be depedent of the OS-sort
        if PREVENT_OS_SORT :
            _path.sort()
            np.random.shuffle(_path)

        #image_list = os.listdir(image_folder)
        # Filtrer pour juste garder les images
        #image_list = [i for i in image_list if '.jpg' in i]
        self.images = np.array([np.array(skiio.imread(image)) for image in _path])
        self.coast_id=[]
        self.forest_id=[]
        self.street_id=[]
        self.class_labels=np.zeros((len(_path),1)).astype('int32')
        for i, name_file in enumerate(_path):
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
        # DO SUB CLASSES
        self.AjoutSubClasses()

        temp = ttsplit( [i for i in range (len(self.class_labels))],
                        self.class_labels, test_size=0.2, shuffle=True)
        self.training_data_idx = temp[0]
        self.training_target = temp[2]
        self.validation_data_idx = temp[1]
        self.validation_target = temp[3]

        #This should be coherent
        dimensions_list = [dimension(name = d_mean_bin,mode = Lab),dimension(name = d_pred_count,mode = Lab)
                           ,dimension(name = d_mean_bin,mode = HSV),dimension(name = d_pred_bin,mode = HSV)]
        self.dims_list=[(d_mean_bin,Lab,1),(d_mean_bin,Lab,2),(d_pred_count,Lab,1)
                        ,(d_mean_bin,HSV,1),(d_pred_bin,HSV,2),(d_pred_bin,HSV,0)]

        self.tracker = VariablesTracker(dimensions_list)
        self.tracker.update_dataset_size(len(self.images))

        self.n_bins=256
        with timeThat('Pre processing of all the data'):
            self.pre_process_all_data(self.images)

        plist =[self.tracker.pick_var(dim[0],dim[1],dim[2]) for dim in self.dims_list]
        self.extent=Extent(ptList=np.stack(plist,axis=1))

        # génération de données aléatoires
        ndonnees = 5000
        self.donneesTest = genDonneesTest(ndonnees, self.extent,n=len(self.dims_list))


    def pre_process_all_data(self,images,n_bins=256):
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

        for i, image in enumerate(images):
            if should_Compute[1]:
                images_HSV = skic.rgb2hsv(image)
                images_HSV = np.round(images_HSV / np.max(images_HSV) * (n_bins - 1)).astype('int32')
                for var in self.tracker :
                    if var.mode==HSV :
                        self.tracker.compute_for_image(images_HSV, i,var)

            if should_Compute[2]:
                # L [0,100],a,b [-127,127]
                images_LAB = skic.rgb2lab(image)
                images_LAB = rescaleHistLab(images_LAB, n_bins)
                images_LAB = images_LAB.astype('int32')
                for var in self.tracker :
                    if var.mode==Lab :
                        self.tracker.compute_for_image(images_LAB, i,var)
            if  should_Compute[0]:
                image = np.round(image / np.max(image) * (n_bins - 1)).astype('int32')
                for var in self.tracker :
                    if var.mode==RGB :
                        self.tracker.compute_for_image(image, i, var)



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

    def get_data_classwise(self,n=.8):

        classes=[ self.all_classes[i][:int(n*len(self.all_classes[i]))] for i in range(N_CLASSES)]
        target=[]
        for i,classe in enumerate(classes ):
            target+=[i]*len(classe)
        target=np.array(target)[:,None]

        val_classes=[ self.all_classes[i][len(classes[i]):] for i in range(N_CLASSES)]

        val_target=[]
        val_idx=[]
        for i,val in enumerate(val_classes):
            val_idx+=val
            val_target+= [i]*len(val)
        val_target=np.array(val_target)[:,None]
        val_data=np.zeros((len(val_idx),len(self.dims_list)))
        data = [ np.zeros((len(classes[i]),len(self.dims_list))) for i in range(N_CLASSES)]
        #data = np.zeros((N_CLASSES, n,len(self.dims_list)))

        for i, dim in enumerate(self.dims_list):
            var = dim[0]
            mode = dim[1]
            index = dim[2]

            temp=self.tracker.pick_var(var, mode, index)
            val_data[:,i]=temp[val_idx]
            for j,class_idx in enumerate(classes):
                data[j][:,i] = temp[class_idx]

        return data,target,val_data,val_target

    def get_target_data(self,train=True):
        return self.class_labels[self.training_data_idx
            if train else self.validation_data_idx]


    def __len__(self):
        return len(self.images)


    def AjoutSubClasses(self):
        dimensions_list = [dimension(name=d_pred_bin, mode=HSV)]
        self.tracker = VariablesTracker(dimensions_list)
        self.tracker.update_dataset_size(len(self.coast_id))
        self.pre_process_all_data(self.images[self.coast_id],6)


        pred_bin = self.tracker.pick_var(dim=d_pred_bin, mode='HSV', index_mode=0)

        coast_sunset = []
        for id, bin in enumerate(pred_bin):
            if bin < 2:
                coast_sunset.append(id)

        self.coast_sunset_id = np.array(self.coast_id)[coast_sunset]
        for indexes in self.coast_sunset_id:
            self.coast_id.remove(indexes)
        self.coast_sunset_id = list(self.coast_sunset_id)
        self.all_classes.append(self.coast_sunset_id)  ## ajout de la coast sunset dans
        ## all classes juste apres coast_id
