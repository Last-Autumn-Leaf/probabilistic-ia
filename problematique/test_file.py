import numpy as np
import numpy.random
import math
import itertools

from ImageCollection import ImageCollection
from helpers.custom_class import dimension, VariablesTracker, getDefaultVar, ClassesTracker
from helpers.custom_helper import *
from helpers.analysis import viewEllipse
import matplotlib.pyplot as plt

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
    #ax = fig.subplots(N_CLASSES, 1)
    axe = fig.subplots(1, 1)
    data = tracker.pick_var(dim, mode, dim_index)

    data = np.round(data / max(data) * n_bin)
    for i in range(N_CLASSES):
        x = data[IC.all_classes[i]]
        y = np.bincount(x.astype('int32'), minlength=n_bin + 1)
        axe.bar(x=[i for i in range(n_bin + 1)], height=y, align='edge', color=CLASS_COLOR_ARRAY[i],alpha=0.5)

def Hist1D(n_bins = 70):
    dimensions_list = [dimension(name=d_mean_bin, mode=Lab), dimension(name=d_mean_bin, mode=HSV),
                       dimension(name=d_fractal, mode=RGB)]
    picked_vars = [(d_mean_bin, Lab, 1), (d_mean_bin, Lab, 2), (d_mean_bin, HSV, 1),
                   (d_fractal, RGB, 0)]

    CT = ClassesTracker(dimensions_list, picked_vars)

    print('We will print', len(picked_vars), 'graphs')
    tracker = CT.tracker
    n_plots_lines = int(np.floor(np.sqrt(len(picked_vars))))
    n_plots_columns = n_plots_lines + 1
    fig, ax = plt.subplots(n_plots_lines, n_plots_columns)

    for i, (dim, mode, mode_index) in enumerate(picked_vars):
        data = tracker.pick_var(dim, mode, mode_index)
        data = np.round(data / max(data) * n_bins)
        for j in range(N_CLASSES):
            x = data[CT.all_classes[j]]
            y = np.bincount(x.astype('int32'), minlength=n_bins + 1)
            ax[i // n_plots_columns, i % n_plots_columns].bar(x=[k for k in range(n_bins + 1)],
                                              height=y, align='edge', color=CLASS_COLOR_ARRAY[j], alpha=0.5)


        ax[i // n_plots_columns, i % n_plots_columns].title.set_text(f'{dim}:{class2detailed_repr[mode][mode_index]}')
    plt.subplots_adjust()
    fig.tight_layout()
    plt.show()

def brute_force_bar(modes=all_modes,vars=all_var_names,n_bins=100) :

    prod = list(itertools.product(modes, vars))
    dimensions_list = [dimension(name=dim, mode=mode) for mode,dim in prod]
    picked_vars=[]
    for m,v in prod :
        if m==RGB and v==d_fractal:
            picked_vars.append ( (v,m,0) )
        elif v==d_fractal :
            pass
        else :
            picked_vars.append((v,m, 0))
            picked_vars.append((v,m, 1))
            picked_vars.append((v,m, 2))


    print('We will print',len(picked_vars),'graphs')
    CT = ClassesTracker(dimensions_list,picked_vars)
    tracker=CT.tracker
    n_plots= int(np.ceil(np.sqrt(len(picked_vars))))
    fig,ax = plt.subplots(n_plots, n_plots,figsize=(50,40))


    for i,(dim, mode, mode_index) in enumerate(picked_vars) :
        data=tracker.pick_var(dim, mode, mode_index)
        data = np.round(data / max(data) * n_bins)
        for j in range(N_CLASSES):
            x = data[CT.all_classes[j]]
            y = np.bincount(x.astype('int32'), minlength=n_bins + 1)
            ax[i // n_plots, i % n_plots].bar(x=[k for k in range(n_bins + 1)],
                        height=y, align='edge', color=CLASS_COLOR_ARRAY[j],alpha=0.5)
        ax[i // n_plots, i % n_plots].title.set_text(f'{dim}:{class2detailed_repr[mode][mode_index]}')
    plt.subplots_adjust()
    fig.tight_layout()
    fig.savefig('bruteforce2.pdf')
    #plt.show()

def graphs3d():
    dimensions_list = [dimension(name=d_mean_bin, mode=Lab), dimension(name=d_mean_bin, mode=HSV),
                       dimension(name=d_fractal, mode=RGB)]
    picked_vars = [(d_mean_bin,Lab,1),(d_mean_bin, Lab, 2),(d_mean_bin, HSV, 1),
                                (d_fractal, RGB, 0)]



    CT = ClassesTracker(dimensions_list, picked_vars)

    picked_vars=list(itertools.combinations(picked_vars,3))
    print('We will print', len(picked_vars), 'graphs')
    tracker = CT.tracker
    n_plots = int(np.ceil(np.sqrt(len(picked_vars))))
    fig=plt.figure(figsize=(50,40))


    for i, (var_x, var_y, var_z) in enumerate(picked_vars):

        x = tracker.pick_var(var_x[0], var_x[1], var_x[2])
        y = tracker.pick_var(var_y[0], var_y[1], var_y[2])
        z = tracker.pick_var(var_z[0], var_z[1], var_z[2])
        ax = fig.add_subplot(n_plots, n_plots, i+1, projection='3d')
        for j in range(N_CLASSES):
            xx = x[CT.all_classes[j]]
            yy = y[CT.all_classes[j]]
            zz = z[CT.all_classes[j]]


            ax.scatter(xx,yy,zz, color=CLASS_COLOR_ARRAY[j], alpha=0.35)
        ax.set_xlabel(f'{var_x[0]}:{class2detailed_repr[var_x[1]][var_x[2]]}')
        ax.set_ylabel(f'{var_y[0]}:{class2detailed_repr[var_y[1]][var_y[2]]}')
        ax.set_zlabel(f'{var_z[0]}:{class2detailed_repr[var_z[1]][var_z[2]]}')
    plt.subplots_adjust()
    fig.tight_layout()

    plt.show()

def graphs2d():
    dimensions_list = [dimension(name=d_mean_bin, mode=Lab), dimension(name=d_mean_bin, mode=HSV),
                       dimension(name=d_fractal, mode=RGB)]
    picked_vars = [(d_mean_bin, Lab, 1), (d_mean_bin, Lab, 2), (d_mean_bin, HSV, 1),
                   (d_fractal, RGB, 0)]

    CT = ClassesTracker(dimensions_list, picked_vars)

    picked_vars = list(itertools.combinations(picked_vars, 2))
    print('We will print', len(picked_vars), 'graphs')
    tracker = CT.tracker
    n_plots_lines = int(np.floor(np.sqrt(len(picked_vars))))
    n_plots_columns  = n_plots_lines + 1
    fig,ax = plt.subplots(n_plots_lines,n_plots_columns)
    for i, (var_x, var_y) in enumerate(picked_vars):

        x = tracker.pick_var(var_x[0], var_x[1], var_x[2])
        y = tracker.pick_var(var_y[0], var_y[1], var_y[2])

        a = i // n_plots_columns
        b = i % n_plots_columns
        for j in range(N_CLASSES):
            xx = x[CT.all_classes[j]]
            yy = y[CT.all_classes[j]]

            ax[a,b].scatter(xx,yy, color=CLASS_COLOR_ARRAY[j], alpha=0.35)

            ax[a, b].scatter(xx, yy, alpha=0.2, color=CLASS_COLOR_ARRAY[j],
                             marker='.')
            data_ellipse = np.array((xx, yy)).T
            viewEllipse(data=data_ellipse, ax=ax[a, b],
                        facecolor=CLASS_COLOR_ARRAY[j], scale=1, alpha=0.5,edgecolor='purple')

        ax[a,b].set_xlabel(f'{var_x[0]}:{class2detailed_repr[var_x[1]][var_x[2]]}')
        ax[a,b].set_ylabel(f'{var_y[0]}:{class2detailed_repr[var_y[1]][var_y[2]]}')
    plt.subplots_adjust()
    fig.tight_layout()

    plt.show()


def main():
    np.random.seed(0)
    for i in range(1):
        np.random.seed(0)
        bar_hist(mode=RGB, dim=d_n_blob, dim_index=i, n_bin=70)
    plt.show()

if __name__ == '__main__':
    # brute_force_bar()
    # main()
    graphs2d()