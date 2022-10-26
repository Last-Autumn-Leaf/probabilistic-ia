

import numpy.random
import math
import itertools

from ImageCollection import plt
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

def AllGraphScatter(IC_obj, mode_list = [Lab,HSV], var_list = [d_mean_bin,d_pred_bin,d_pred_count],n_bins=256):
    colors = ['blue', 'green', 'black']



    ### Taille du subplot generaliste ###
    nb_mode = len(mode_list)
    nb_var = len(var_list)
    nb_comb = 3 * nb_var * nb_mode
    nb_plot = math.comb(nb_comb, 2)
    length_plot = int(np.ceil(np.sqrt(nb_plot)))
    fig, axs = plt.subplots(length_plot,length_plot, figsize = (30,10))
    ### --- ###

    dimensions_list =[dimension(name = dim,mode = mode) for mode in mode_list for dim in var_list]
    tracker = VariablesTracker(dimensions_list)


    tracker.update_dataset_size(len(IC_obj.image_list))

    IC_obj.getStat([i for i in range(len(IC_obj.image_list))], tracker, n_bins=n_bins)


    couple_dim = list(itertools.product(mode_list, var_list))
    idx_graphe = 0

    nb_graph_sqrt = len(couple_dim)
    # nb_graph_sqrt = 1
    for couple1 in range(nb_graph_sqrt):
        for couple2 in range(couple1,nb_graph_sqrt):
            if couple1 == couple2:
                for ax1_param in range(3):
                    for ax2_param in range(ax1_param+1,3):

                        var1 = couple_dim[couple1][1]
                        mode1 = couple_dim[couple1][0]
                        index1 = ax1_param

                        var2 = couple_dim[couple2][1]
                        mode2 = couple_dim[couple2][0]
                        index2 = ax2_param



                        for idx_class, c_class in enumerate(IC_obj.all_classes):
                            x = tracker.pick_var(var1, mode1, index1)[c_class]
                            y = tracker.pick_var(var2, mode2, index2)[c_class]

                            axs[idx_graphe//length_plot,idx_graphe%length_plot].scatter(x, y, alpha=0.4, color=colors[idx_class], marker='.')
                        print(f"graphe {idx_graphe}")
    # x.set_title('')

                        # x.set_title('')
                        #
                        axs[idx_graphe//length_plot,idx_graphe%length_plot].set(xlabel=f"{class2detailed_repr[mode1][index1]}({var1})",
                               ylabel=f"{class2detailed_repr[mode2][index2]}({var2})")
                        idx_graphe += 1
        # for mode,var in couple_dim:

            else:
                for ax1_param in range(3):
                    for ax2_param in range(3):

                        var1 = couple_dim[couple1][1]
                        mode1 = couple_dim[couple1][0]
                        index1 = ax1_param

                        var2 = couple_dim[couple2][1]
                        mode2 = couple_dim[couple2][0]
                        index2 = ax2_param


                        for idx_class, c_class in enumerate(IC_obj.all_classes):
                            x = tracker.pick_var(var1, mode1, index1)[c_class]
                            y = tracker.pick_var(var2, mode2, index2)[c_class]

                            axs[idx_graphe//length_plot,idx_graphe%length_plot].scatter(x, y, alpha=0.4, color=colors[idx_class],
                                                                                  marker='.')
                        print(f"graphe {idx_graphe}")

                        # x.set_title('')
                        #
                        axs[idx_graphe//length_plot,idx_graphe%length_plot].set(xlabel=f"{class2detailed_repr[mode1][index1]}({var1})",
                               ylabel=f"{class2detailed_repr[mode2][index2]}({var2})")
                        idx_graphe += 1
            # for mode,var in couple_dim:
    plt.subplots_adjust()
    fig.tight_layout()

    fig.savefig('bruteforce.pdf', bbox_inches='tight')



def main():
    np.random.seed(0)
    IC =ImageCollection()

    ###
    mode_list = [Lab,HSV]
    var_list = [d_pred_bin]
    ###

    mode_scatter1 = HSV
    mode_scatter2 = HSV
    dim_scatter1 = d_pred_bin
    dim_scatter2 = d_pred_bin
    dimensions_list =[dimension(name = dim_scatter1,mode = mode_scatter1),dimension(name = dim_scatter2,mode = mode_scatter2)]
    tracker = VariablesTracker(dimensions_list)


    d1=(dim_scatter1,mode_scatter1,0)
    d2=(dim_scatter2,mode_scatter2,2)

    with timeThat() :
        IC.scatterGraph2D(d1,d2,tracker,n_bins=256)
        AllGraphScatter(IC_obj=IC,mode_list=mode_list,var_list=var_list,n_bins=256)

    plt.show()


if __name__ == '__main__':
    main()