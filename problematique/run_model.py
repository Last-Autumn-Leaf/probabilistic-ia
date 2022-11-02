import os.path

from helpers import classifiers
from helpers.custom_class import *
import matplotlib.pyplot  as plt
from helpers.custom_class import ClassesTracker

def plotALlConfusionMaxtixes ():

    def createConfusionMatrix(pred,target):
        cf_m=confusion_matrix(pred,target)
        plot_confusion_matrix(cf_m)

    foo_list = [Bayes, KNN, RNN]
    for i, foo in enumerate(foo_list):
        score, pred, target = foo()
        createConfusionMatrix(pred, target)

    plt.show()

def Bayes(n=0.83):
    np.random.seed(0)
    train_data, train_classes, test_data, test_classes = CT.get_data_classwise(n=n)
    score,pred,target= classifiers.full_Bayes_risk(train_data, train_classes, CT.donneesTest, 'Bayes risque #1',
                                CT.extent, test_data, test_classes)
    return score,pred,target

def KNN(n_kmean=N_KMEAN,n_knn=N_KNN,cluster_centers=None,cluster_labels=None,plot=True):
    train_data, train_classes, test_data, test_classes = CT.get_data_classwise(n=0.84)
    if cluster_centers == None or cluster_labels==None :
        if os.path.exists(KNN_MODEL_PATH[0]) and os.path.exists(KNN_MODEL_PATH[1]) :
            cluster_centers, cluster_labels =np.load(KNN_MODEL_PATH[0]),np.load(KNN_MODEL_PATH[1])
        else:
            print('no clusters found')
            cluster_centers, cluster_labels = classifiers.full_kmean(n_kmean, train_data,
        train_classes,'Représentants des ' + f'{n_kmean}' + '-moy',CT.extent,plot=plot)
    score,pred,target= classifiers.full_ppv(n_knn, cluster_centers, cluster_labels, CT.donneesTest,
                         f'{n_knn}-PPV sur le ' + f'{n_kmean}' + '-moy', CT.extent, test_data,
                         test_classes,plot=plot)
    return score, pred, target

def RNN():
    np.random.seed(0)
    Quatres_data = CT.get_all_data()
    Quatres_labels = CT.class_labels
    test_data, test_classes = CT.get_data_classwise(n=0.80)[2:4]

    ### Hyperparametres ###
    n_hidden_layers = HIDDEN_LAYERS
    n_neurons = N_NEURONS
    ### --- ###

    score, pred, target=classifiers.full_nn(n_hiddenlayers = n_hidden_layers, n_neurons= n_neurons, train_data= Quatres_data,
            train_classes=Quatres_labels, test1=CT.donneesTest, title =f'NN {n_hidden_layers} layer(s) caché(s), {n_neurons} neurones par couche',
                                extent = CT.extent,test2 = test_data,classes2=test_classes )
    return score, pred, target


# -------- recherche d'hyper-paramètres :
def param_search_KNN(it=100):
    max_knn_score = 0
    with timeThat(f'Best knn score'):
        for n_kmean in range(4, 20):
            print('n_kmean',n_kmean)
            for n_knn in range(1, n_kmean, 2):
                for i in range(it):
                    if int(i/it*100) %20==0: print(int(i / it*100), '%')
                    c_score,knn_data = KNN(n_kmean,n_knn,plot=False)
                    if c_score > max_knn_score:
                        max_knn_score = c_score
                        print('storing for',c_score,n_kmean,n_knn)
                        np.save(KNN_MODEL_PATH[0],knn_data[0])
                        np.save(KNN_MODEL_PATH[1],knn_data[1])


np.random.seed(0)
CT = ClassesTracker()

if __name__=='__main__':
    plotALlConfusionMaxtixes()
