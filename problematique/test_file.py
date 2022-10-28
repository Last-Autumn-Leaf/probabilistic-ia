

import numpy.random
import math
import itertools

from ImageCollection import ImageCollection
from helpers.custom_class import dimension, VariablesTracker, getDefaultVar
from helpers.custom_helper import *
from helpers.analysis import viewEllipse



def AllGraphScatter(IC_obj, mode_list = [Lab,HSV], var_list = [d_mean_bin,d_pred_bin,d_pred_count],n_bins=256):
    IC_obj.AjoutSubClasses()
    colors = CLASS_COLOR_ARRAY

    ### Taille du subplot generaliste ###
    nb_mode = len(mode_list)-1
    nb_var = len(var_list)
    nb_comb = 3 * nb_var * nb_mode
    nb_plot = math.comb(nb_comb, 2)
    length_plot = int(np.ceil(np.sqrt(nb_plot)))
    fig, axs = plt.subplots(length_plot,length_plot, figsize = (50,40))
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

                        if (var1 ==d_fractal and (mode1!=RGB or index1!=0) ) or \
                                (var2 ==d_fractal and (mode2!=RGB and index2!=0) ):
                            ...
                        elif mode1==RGB and var1!=d_fractal or mode2==RGB and var2!=d_fractal:
                            ...
                        else :

                            for idx_class, c_class in enumerate(IC_obj.all_classes):
                                x = tracker.pick_var(var1, mode1, index1)[c_class]
                                y = tracker.pick_var(var2, mode2, index2)[c_class]

                                axs[idx_graphe//length_plot,idx_graphe%length_plot].scatter(x, y, alpha=0.2, color=colors[idx_class], marker='.')
                                data_ellipse = np.array((x, y)).T
                                viewEllipse(data=data_ellipse, ax=axs[idx_graphe//length_plot,idx_graphe%length_plot], facecolor=colors[idx_class], scale=1, alpha=0.25)

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

                        if (var1 ==d_fractal and (mode1!=RGB or index1!=0) ) or \
                                (var2 ==d_fractal and (mode2!=RGB and index2!=0) ):
                            ...
                        elif mode1==RGB and var1!=d_fractal or mode2==RGB and var2!=d_fractal:
                            ...
                        else :

                            for idx_class, c_class in enumerate(IC_obj.all_classes):
                                x = tracker.pick_var(var1, mode1, index1)[c_class]
                                y = tracker.pick_var(var2, mode2, index2)[c_class]

                                axs[idx_graphe//length_plot,idx_graphe%length_plot].scatter(x, y, alpha=0.2, color=colors[idx_class],
                                                                                      marker='.')
                                data_ellipse = np.array((x, y)).T
                                viewEllipse(data = data_ellipse,ax = axs[idx_graphe//length_plot,idx_graphe%length_plot], facecolor = colors[idx_class],scale= 1,alpha=0.25)

                            print(f"graphe {idx_graphe}")

                            # x.set_title('')
                            #
                            axs[idx_graphe//length_plot,idx_graphe%length_plot].set(xlabel=f"{class2detailed_repr[mode1][index1]}({var1})",
                                   ylabel=f"{class2detailed_repr[mode2][index2]}({var2})")
                            idx_graphe += 1
            # for mode,var in couple_dim:
    plt.subplots_adjust()
    fig.tight_layout()

    fig.savefig('bruteforce.pdf')
    plt.show()

def bar_hist(mode=RGB,dim=d_mean_bin,dim_index=0,n_bin=20):
    IC = ImageCollection()
    IC.AjoutSubClasses()

    dimensions_list = [dimension(name=dim, mode=mode)]
    tracker = VariablesTracker(dimensions_list)

    with timeThat('Get Stats'):
        IC.getStat([i for i in range(980)], tracker)

    fig = plt.figure()
    fig.suptitle(f'Bar histogram {dim}', fontsize=20)
    ax = fig.subplots(N_CLASSES, 1)
    data = tracker.pick_var(dim, mode, dim_index)
    data = np.round(data / max(data) * n_bin)
    for i, axe in enumerate(ax):
        x = data[IC.all_classes[i]]
        y = np.bincount(x.astype('int32'), minlength=n_bin + 1)
        axe.bar(x=[i for i in range(n_bin + 1)], height=y, align='edge', color=CLASS_COLOR_ARRAY[i])

    plt.show()
def main():
    np.random.seed(0)
    # AllGraphScatter(IC_obj=ImageCollection(),mode_list=[RGB,Lab,HSV],var_list=all_var_names)
    bar_hist(mode=RGB,dim=d_fractal,dim_index=0,n_bin=20)

if __name__ == '__main__':
    main()