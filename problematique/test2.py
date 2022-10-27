import numpy as np

from helpers import classifiers
import matplotlib.pyplot  as plt
from helpers.custom_class import ClassesTracker

if __name__=='__main__':
    np.random.seed(0)
    CT = ClassesTracker()
    train_data,train_classes, test_data, test_classes=CT.get_data_classwise(n=0.8)
    classifiers.full_Bayes_risk(train_data, train_classes, CT.donneesTest, 'Bayes risque #1',
                                CT.extent, test_data, test_classes)

    plt.show()




