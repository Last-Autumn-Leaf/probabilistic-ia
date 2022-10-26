from helpers import classifiers
from helpers.custom_helper import *
import matplotlib.pyplot  as plt
if __name__=='__main__':

    CT = ClassesTracker()
    train_data,train_classes, test_data, test_classes=CT.get_data_classwise()
    classifiers.full_Bayes_risk(train_data, train_classes, CT.donneesTest, 'Bayes risque #1',
                                CT.extent, test_data, test_classes)

    plt.show()




