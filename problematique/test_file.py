import numpy.random

from ImageCollection import plt
import numpy as np
import random
from ImageCollection import ImageCollection


get_n_rand = lambda sett=ImageCollection.coast_id,n=1 :random.sample(sett, n)
def main():
    numpy.random.seed(0)

    IC =ImageCollection()
    idx=IC.forest_id[:5]
    #IC.getDatasetTable()
    IC.getDatasetTable(current_mode='Lab',n_bins=6)

    #IC.view_histogrammes(IC.coast_id[0])
    #IC.images_display(idx)

    plt.show()


if __name__ == '__main__':
    main()