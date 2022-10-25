import time
from datetime import  timedelta

import numpy.random

from ImageCollection import plt
import numpy as np
from ImageCollection import ImageCollection
from helpers.custom_helper import *

def test_track_list(tracker):
        for var in tracker.variables  :
            print('________________________')
            print(var)
            print('\tdata:\n',var.data) # les données brutes de taille (3 ou 4) x datase_size
            if var.isAvg :
                print('\tmean:\n',var.mean) #les données moyennées de taille (3 ou 4) x 1
            else:
                print('No mean')


def main():
    np.random.seed(0)
    IC =ImageCollection()


    # Ici on choisit les dimensions que l'on veut surveillé et calculer !
    dim_name=[d_pred_bin,d_pred_count]
    # petite fonction our initiliser avec les paramètres par défaut les dimensions
    # Pour L'instant il n'y a qu'un seul paramètre qui est si on calcul la moyenne ou non !
    # Donc pas vraiment utile.
    # La classe dimension a besoin de connaitre la taille du dataset afin de pré-allouer la mémoire
    mode_list = [RGB,Lab,HSV]
    dimensions_list =[dimension(name = dim_name[1],mode = mode_list[1]),dimension(name = dim_name[0],mode = mode_list[1])]
    tracker = VariablesTracker(dimensions_list)

    d1=(d_pred_bin,mode_list[1],0)
    d2=(d_pred_count,mode_list[1],0)

    with timeThat() :
        ...
        IC.scatterGraph2D(d1,d2,tracker,n_bins=64)

    # M1 = IC.GetCovMatrix(dimension_parameters = dimensions_list,tracker = tracker, n_bins=256)
    #IC.scatterGraph2D

    plt.show()


from contextlib import contextmanager
@contextmanager
def timeThat(name=''):
    try:
        start = time.time()
        yield ...
    finally:
        end = time.time()
        print(name+' finished in ',timedelta(seconds=end-start))


if __name__ == '__main__':
    main()