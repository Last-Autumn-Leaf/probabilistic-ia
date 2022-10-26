

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


    mode_scatter1 = HSV
    mode_scatter2 = HSV
    dim_scatter1 = d_pred_bin
    dim_scatter2 = d_pred_bin
    dimensions_list =[dimension(name = dim_scatter1,mode = mode_scatter1),dimension(name = dim_scatter2,mode = mode_scatter2)]
    tracker = VariablesTracker(dimensions_list)

    d1=(dim_scatter1,mode_scatter1,2)
    d2=(dim_scatter2,mode_scatter2,1)

    with timeThat() :
        ...
        IC.scatterGraph2D(d1,d2,tracker,n_bins=256)

    # M1 = IC.GetCovMatrix(dimension_parameters = dimensions_list,tracker = tracker, n_bins=256)
    #IC.scatterGraph2D

    plt.show()





if __name__ == '__main__':
    main()