import numpy.random

from ImageCollection import plt
import numpy as np
from ImageCollection import ImageCollection
from helpers.custom_helper import *


def main():
    np.random.seed(0)
    IC =ImageCollection()

    #Choix du Dataset
    dataset_size=5
    idx = get_n_rand_from_set(IC.coast_id, dataset_size)

    # Ici on choisit les dimensions que l'on veut surveillé et calculer !
    dim_name=[d_mean_bin,v_pred_bin]
    # petite fonction our initiliser avec les paramètres par défaut les dimensions
    # Pour L'instant il n'y a qu'un seul paramètre qui est si on calcul la moyenne ou non !
    # Donc pas vraiment utile.
    # La classe dimension a besoin de connaitre la taille du dataset afin de pré-allouer la mémoire

    dims =[getDefaultVar(name, dataset_size) for name in dim_name]
    tracker= VariablesTracker(dims)

    # Les calculs se font dans cette fonction
    IC.getStat(idx,tracker)


    for var in tracker.variables :
        print('________________________')
        print(var)
        print('\tdata:\n',var.data) # les données brutes de taille (3 ou 4) x datase_size
        if var.isAvg :
            print('\tmean:\n',var.mean) #les données moyennées de taille (3 ou 4) x 1
        else:
            print('No mean')
    #plt.show()


if __name__ == '__main__':
    main()