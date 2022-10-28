import numpy as np

import helpers.custom_helper

from helpers import classifiers
from helpers.custom_class import *
import matplotlib.pyplot  as plt
from helpers.custom_class import ClassesTracker


def Bayes():
    CT = ClassesTracker()
    train_data, train_classes, test_data, test_classes = CT.get_data_classwise(n=0.8)
    classifiers.full_Bayes_risk(train_data, train_classes, CT.donneesTest, 'Bayes risque #1',
                                CT.extent, test_data, test_classes)




def KNN(n_kmean=5,n_knn=1):

    CT = ClassesTracker()
    train_data, train_classes, test_data, test_classes = CT.get_data_classwise(n=0.8)
    cluster_centers, cluster_labels = classifiers.full_kmean(n_kmean, train_data,
        train_classes,'Représentants des ' + f'{n_kmean}' + '-moy',CT.extent)


    classifiers.full_ppv(n_knn, cluster_centers, cluster_labels, CT.donneesTest,
                         f'{n_knn}-PPV sur le ' + f'{n_kmean}' + '-moy', CT.extent, test_data,
                         test_classes)



def RNN():

    CT = ClassesTracker()
    Quatres_data = CT.get_all_data()
    Quatres_labels = CT.class_labels

    ### Hyperparametres ###
    n_hidden_layers = 2
    n_neurons = 5
    ### --- ###


    # train_data, train_classes, test_data, test_classes = CT.get_data_classwise(n=0.8)
    classifiers.full_nn(n_hiddenlayers = n_hidden_layers, n_neurons= n_neurons, train_data= Quatres_data, train_classes=Quatres_labels, test1=CT.donneesTest, title =f'NN {n_hidden_layers} layer(s) caché(s), {n_neurons} neurones par couche',
                                extent = CT.extent,test2 = Quatres_data,classes2=Quatres_labels )
# test_data, test_classes
if __name__=='__main__':
    np.random.seed(0)
    # for thresh in range(98, 103):
    #     helpers.custom_helper.fractal_thr = thresh
    #     Bayes()
    # KNN()
    RNN()

    plt.show()




