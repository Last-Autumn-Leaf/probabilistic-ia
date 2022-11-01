import os.path

import helpers.custom_helper

from helpers import classifiers
from helpers.custom_class import *
import matplotlib.pyplot  as plt
from helpers.custom_class import ClassesTracker


def Bayes(n=0.83):
    np.random.seed(0)
    train_data, train_classes, test_data, test_classes = CT.get_data_classwise(n=n)
    return classifiers.full_Bayes_risk(train_data, train_classes, CT.donneesTest, 'Bayes risque #1',
                                CT.extent, test_data, test_classes)

def KNN(n_kmean=39,n_knn=5):
    np.random.seed(0)
    train_data, train_classes, test_data, test_classes = CT.get_data_classwise(n=0.84)
    cluster_centers, cluster_labels = classifiers.full_kmean(n_kmean, train_data,
        train_classes,'Représentants des ' + f'{n_kmean}' + '-moy',CT.extent,plot=False)
    return classifiers.full_ppv(n_knn, cluster_centers, cluster_labels, CT.donneesTest,
                         f'{n_knn}-PPV sur le ' + f'{n_kmean}' + '-moy', CT.extent, test_data,
                         test_classes)

def RNN():
    np.random.seed(0)
    Quatres_data = CT.get_all_data()
    Quatres_labels = CT.class_labels


    ### Hyperparametres ###
    n_hidden_layers = 7
    n_neurons = 5
    ### --- ###

    classifiers.full_nn(n_hiddenlayers = n_hidden_layers, n_neurons= n_neurons, train_data= Quatres_data, train_classes=Quatres_labels, test1=CT.donneesTest, title =f'NN {n_hidden_layers} layer(s) caché(s), {n_neurons} neurones par couche',
                                extent = CT.extent,test2 = Quatres_data,classes2=Quatres_labels )



# -------- recherche d'hyper-paramètres :
def param_search_BAYES():
    max_bayes = 0
    with timeThat(f'Best Bayes score'):
        for n in range(70, 95, 1):
            c_score = Bayes(n / 100)
            if c_score > max_bayes:
                max_bayes = c_score
                print(f'BEST PARAMS {np.round(max_bayes, 2)}%', 'n=', n / 100)

def param_search_KNN():
    max_knn_score = 0
    with timeThat(f'Best knn score'):
        for n_kmean in range(1, 40):
            print(n_kmean / 40, '%')
            for n_knn in range(1, n_kmean, 2):
                c_score = KNN(n_kmean=n_kmean, n_knn=n_knn)
                if c_score > max_knn_score:
                    max_knn_score = c_score
                    print(f'BEST PARAMS {np.round(max_knn_score, 2)}%', 'kmean=', n_kmean, 'KNN=', n_knn)



np.random.seed(0)
CT = ClassesTracker()

if __name__=='__main__':

    Bayes()
    KNN()
    RNN()


