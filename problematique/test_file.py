import numpy.random

from ImageCollection import plt
import numpy as np
import random
from ImageCollection import ImageCollection



get_n_rand = lambda sett=ImageCollection.coast_id,n=1 :random.sample(sett, n)
def main():
    np.random.seed(0)


    IC =ImageCollection()
    idx=IC.forest_id[:5]
    IC.getDatasetScatterGraph('RGB',n_bins=32,watch = IC.watch_var[3])
    IC.getDatasetTable(current_mode='RGB',n_bins = 32, watch = IC.watch_var[3])

    #IC.view_histogrammes(IC.coast_id[0])
    #IC.images_display(idx)

    plt.show()


if __name__ == '__main__':
    main()