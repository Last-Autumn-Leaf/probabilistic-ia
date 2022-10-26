from helpers import classifiers
from helpers.custom_helper import *

if __name__=='__main__':

    CT = ClassesTracker()
    a=CT.get_training_data()

    classifiers.full_Bayes_risk(CT.get_training_data(), CT.training_target, CT.get_test_data(), 'Bayes risque #1',
                                CT.extent, CT.get_test_data(), CT.get_test_data())


    # Bayes test
    # (train_data, train_classes, donnee_test, title, extent, test_data, test_classes)
    #classifiers.full_Bayes_risk(allClasses, TroisClasses.class_labels, donneesTest, 'Bayes risque #1',
    #                            TroisClasses.extent, TroisClasses.data, TroisClasses.class_labels)

    # train data : liste de points c*n*m de dimension c le nombre de classe n nombre d'images m nombres de dimensions
    # train_classes c*n
    # donnee_test o*m
    # test class


