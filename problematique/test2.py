import numpy as np

import helpers.custom_helper

np.random.seed(0)
from helpers import classifiers
from helpers.custom_class import *
import matplotlib.pyplot  as plt
from helpers.custom_class import ClassesTracker


def Bayes():
    CT = ClassesTracker()
    train_data, train_classes, test_data, test_classes = CT.get_data_classwise(n=0.8)
    classifiers.full_Bayes_risk(train_data, train_classes, CT.donneesTest, 'Bayes risque #1',
                                CT.extent, test_data, test_classes)




def KNN():
    CT = ClassesTracker()
    train_data, train_classes, test_data, test_classes = CT.get_data_classwise(n=0.8)
    classifiers.full_Bayes_risk(train_data, train_classes, CT.donneesTest, 'Bayes risque #1',
                                CT.extent, test_data, test_classes)
    plt.show()


if __name__=='__main__':
    np.random.seed(0)

    for thresh in range(98, 103):
        helpers.custom_helper.fractal_thr = thresh
        Bayes()




