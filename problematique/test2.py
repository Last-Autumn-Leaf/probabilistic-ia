import random

import numpy as np

import helpers.custom_helper

from helpers import classifiers
from helpers.custom_class import *
import matplotlib.pyplot  as plt
from helpers.custom_class import ClassesTracker


def Bayes(n=0.84):
    train_data, train_classes, test_data, test_classes = CT.get_data_classwise(n=n)
    return classifiers.full_Bayes_risk(train_data, train_classes, CT.donneesTest, 'Bayes risque #1',
                                CT.extent, test_data, test_classes,plot=False)




def KNN(n_kmean=19,n_knn=11):


    train_data, train_classes, test_data, test_classes = CT.get_data_classwise(n=0.94)
    cluster_centers, cluster_labels = classifiers.full_kmean(n_kmean, train_data,
        train_classes,'Représentants des ' + f'{n_kmean}' + '-moy',CT.extent,plot=False)
    return classifiers.full_ppv(n_knn, cluster_centers, cluster_labels, CT.donneesTest,
                         f'{n_knn}-PPV sur le ' + f'{n_kmean}' + '-moy', CT.extent, test_data,
                         test_classes,plot=False)



def RNN():
    CT = ClassesTracker()
    train_data, train_classes, test_data, test_classes = CT.get_data_classwise(n=0.8)
    classifiers.full_nn(train_data, train_classes, CT.donneesTest, ' RNN',
                                CT.extent, test_data, test_classes)

np.random.seed(0)
CT = ClassesTracker()
if __name__=='__main__':


    max_bayes=0
    with timeThat(f'Best knn score'):
        for n_kmean in range(1,20):
            print(n_kmean/20,'%')
            for n_knn in range(1,n_kmean,2):
                    c_score=KNN(n_kmean=n_kmean,n_knn=n_knn)
                    if c_score>max_bayes :
                        max_bayes=c_score
                        print(f'BEST PARAMS {np.round(max_bayes,2)}%','kmean=',n_kmean,'KNN=',n_knn)

    max_bayes=0
    with timeThat(f'Best Bayes score'):
        for n in range(70, 95,1):
            c_score=Bayes(n/100)
            if c_score > max_bayes:
                max_bayes = c_score
                print(f'BEST PARAMS {np.round(max_bayes,2)}%', 'n=', n/100)

    #plt.show()




