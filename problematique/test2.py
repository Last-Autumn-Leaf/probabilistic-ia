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




if __name__=='__main__':
    np.random.seed(0)
    Bayes()
    KNN()

    plt.show()




