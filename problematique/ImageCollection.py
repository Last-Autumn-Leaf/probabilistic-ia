"""
Classe "ImageCollection" statique pour charger et visualiser les images de la problématique
Membres statiques:
    image_folder: le sous-répertoire d'où les images sont chargées
    image_list: une énumération de tous les fichiers .jpg dans le répertoire ci-dessus
    images: une matrice de toutes les images, (optionnelle, décommenter le code)
    all_images_loaded: un flag qui indique si la matrice ci-dessus contient les images ou non
Méthodes statiques: TODO JB move to helpers
    images_display: affiche quelques images identifiées en argument
    view_histogrammes: affiche les histogrammes de couleur de qq images identifiées en argument
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from skimage import color as skic
from skimage import io as skiio
import matplotlib.cm as cm
import pandas as pd

from helpers.analysis import viewEllipse
from helpers.custom_helper import getHighestFrequencyVector


class ImageCollection:
    """
    Classe globale pour regrouper les infos utiles et les méthodes de la collection d'images
    """

    # liste de toutes les images
    image_folder = r"." + os.sep + "baseDeDonneesImages"
    _path = glob.glob(image_folder + os.sep + r"*.jpg")
    image_list = os.listdir(image_folder)
    # Filtrer pour juste garder les images
    image_list = [i for i in image_list if '.jpg' in i]


    all_images_loaded = False
    images = []

    # # Créer un array qui contient toutes les images
    # # Dimensions [980, 256, 256, 3]
    # #            [Nombre image, hauteur, largeur, RGB]
    # # TODO décommenter si voulu pour charger TOUTES les images
    # images = np.array([np.array(skiio.imread(image)) for image in _path])
    # all_images_loaded = True

    # Custom addings :

    # no use keeping in memory all the images

    # we want to seperate the 3 classes :
    coast_id=[]
    forest_id=[]
    street_id=[]
    for i,name_file in  enumerate(image_list) :
        if "coast" in name_file :
            coast_id.append(i)
        elif "forest" in name_file :
            forest_id.append(i)
        else  :
            street_id.append(i)

    # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
    n_bins = 256  #

    enc_classes = {'coast': coast_id,
        'forest':forest_id,
        'street':street_id}
    all_classes=[k for k in enc_classes]


    def images_display(self,indexes):
        """
        fonction pour afficher les images correspondant aux indices
        indexes: indices de la liste d'image (int ou list of int)
        """

        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig2 = plt.figure()
        ax2 = fig2.subplots(len(indexes), 1)

        for i in range(len(indexes)):
            if ImageCollection.all_images_loaded:
                im = ImageCollection.images[i]
            else:
                im = skiio.imread(ImageCollection.image_folder + os.sep + ImageCollection.image_list[indexes[i]])
            ax2[i].imshow(im) if len(indexes)!=1 else ax2.imshow(im)

    # helper function pour rescaler le format lab
    def rescaleHistLab(self,LabImage, n_bins):
        """
        Helper function
        La représentation Lab requiert un rescaling avant d'histogrammer parce que ce sont des floats!
        """
        # Constantes de la représentation Lab
        class LabCte:      # TODO JB : utiliser an.Extent?
            min_L: int = 0
            max_L: int = 100
            min_ab: int = -110
            max_ab: int = 110
        # Création d'une image vide
        imageLabRescale = np.zeros(LabImage.shape)
        # Quantification de L en n_bins niveaux
        imageLabRescale[:, :, 0] = np.round(
            (LabImage[:, :, 0] - LabCte.min_L) * (n_bins - 1) / (
                    LabCte.max_L - LabCte.min_L))  # L has all values between 0 and 100
        # Quantification de a et b en n_bins niveaux
        imageLabRescale[:, :, 1:3] = np.round(
            (LabImage[:, :, 1:3] - LabCte.min_ab) * (n_bins - 1) / (
                    LabCte.max_ab - LabCte.min_ab))  # a and b have all values between -110 and 110
        return imageLabRescale


    def view_histogrammes(self,indexes):
        """
        Affiche les histogrammes de couleur de quelques images
        indexes: int or list of int des images à afficher
        """
        ###########################################
        # view_histogrammes starts here
        ###########################################
        # TODO JB split calculs et view en 2 fonctions séparées
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig = plt.figure()
        fig.suptitle('RGB,Lab', fontsize=20)
        ax = fig.subplots(len(indexes), 2)
        if len(indexes)==1 :
            ax=ax[None,:]

        for num_images in range(len(indexes)):
            # charge une image si nécessaire
            if ImageCollection.all_images_loaded:
                imageRGB = ImageCollection.images[num_images]
            else:
                imageRGB = skiio.imread(
                    ImageCollection.image_folder + os.sep + ImageCollection.image_list[indexes[num_images]])

            # Exemple de conversion de format pour Lab et HSV
            imageLab = skic.rgb2lab(imageRGB)  # TODO L1.E3.5: afficher ces nouveaux histogrammes
            imageHSV = skic.rgb2hsv(imageRGB)  # TODO problématique: essayer d'autres espaces de couleur

            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = ImageCollection.n_bins

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            imageLabhist = self.rescaleHistLab(imageLab, n_bins) # External rescale pour Lab
            imageHSVhist = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100

            # Construction des histogrammes
            # 1 histogram per color channel
            pixel_valuesRGB = np.zeros((3, n_bins))
            pixel_valuesLab = np.zeros((3, n_bins))
            pixel_valuesHSV = np.zeros((3, n_bins))
            for i in range(n_bins):
                for j in range(3):
                    pixel_valuesRGB[j, i] = np.count_nonzero(imageRGB[:, :, j] == i)
                    pixel_valuesLab[j, i] = np.count_nonzero(imageLabhist[:, :, j] == i)
                    pixel_valuesHSV[j, i] = np.count_nonzero(imageHSVhist[:, :, j] == i)

            # permet d'omettre les bins très sombres et très saturées aux bouts des histogrammes
            skip = 5
            start = skip
            end = n_bins - skip

            # affichage des histogrammes
            ax[num_images, 0].scatter(range(start, end), pixel_valuesRGB[0, start:end], c='red')
            ax[num_images, 0].scatter(range(start, end), pixel_valuesRGB[1, start:end], c='green')
            ax[num_images, 0].scatter(range(start, end), pixel_valuesRGB[2, start:end], c='blue')
            #ax[num_images, 0].set(xlabel='pixels', ylabel='compte par valeur d\'intensité')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = ImageCollection.image_list[indexes[num_images]]
            #ax[num_images, 0].set_title(f'histogramme RGB de {image_name}')

            # 2e histogramme
            # TODO L1.E3 afficher les autres histogrammes de Lab ou HSV dans la 2e colonne de subplots
            ax[num_images, 1].scatter(range(start, end), pixel_valuesLab[0, start:end], c='black')
            ax[num_images, 1].scatter(range(start, end), pixel_valuesLab[1, start:end], c='red')
            ax[num_images, 1].scatter(range(start, end), pixel_valuesLab[2, start:end], c='blue')
            #ax[num_images, 1].set(xlabel='pixels', ylabel='compte par valeur d\'intensité')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = ImageCollection.image_list[indexes[num_images]]
            #ax[num_images, 1].set_title(f'histogramme Lab de {image_name}')

    def view_HSV_histogram(self,indexes,n_bins_H=6):
        if type(indexes) == int:
            indexes = [indexes]

        fig = plt.figure()
        fig.suptitle('HSV representation', fontsize=20)
        ax = fig.subplots(len(indexes),4)
        if len(indexes) ==1:
            ax=ax[None,:]

        for num_images in range(len(indexes)):
            # charge une image si nécessaire
            if ImageCollection.all_images_loaded:
                imageRGB = ImageCollection.images[num_images]
            else:
                imageRGB = skiio.imread(
                    ImageCollection.image_folder + os.sep + ImageCollection.image_list[indexes[num_images]])

            imageHSV = skic.rgb2hsv(imageRGB)  # TODO problématique: essayer d'autres espaces de couleur

            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = self.n_bins

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            imageHSVhist = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100

            imageHSVhist_2 = np.round(imageHSV * (n_bins_H - 1))
            pixel_val_hist = np.zeros(n_bins_H)

            pixel_valuesHSV = np.zeros((3, n_bins))
            for i in range(n_bins):
                for j in range(3):
                    pixel_valuesHSV[j, i] = np.count_nonzero(imageHSVhist[:, :, j] == i)

            for i in range(n_bins_H):
                pixel_val_hist[ i] = np.count_nonzero(imageHSVhist_2[:, :, 0] == i)

            # permet d'omettre les bins très sombres et très saturées aux bouts des histogrammes
            skip = 5
            start = skip
            end = n_bins - skip

            ax[num_images,0].scatter(range(start, end), pixel_valuesHSV[0, start:end],s=5, c=range(start, end),cmap='hsv')
            hsv_colors = cm.hsv(np.linspace(0,1,n_bins_H))
            max_bar=360
            ax[num_images,1].bar(x=[max_bar/n_bins_H*i for i in range(n_bins_H)],height=pixel_val_hist,
                                 align='edge',width=max_bar/n_bins_H,color=hsv_colors)

            ax[num_images,2].scatter(range(start, end), pixel_valuesHSV[1, start:end],s=5, c='k') # sombre-clair
            ax[num_images,3].scatter(range(start, end), pixel_valuesHSV[2, start:end],s=5, c='k')
            image_name = ImageCollection.image_list[indexes[num_images]]

    def getStat(self,indexes,tracker,mode='RGB',n_bins=256):

        if type(indexes) == int:
            indexes = [indexes]
        for num_images in range(len(indexes)):
            # charge une image si nécessaire
            if ImageCollection.all_images_loaded:
                images = ImageCollection.images[num_images]
            else:
                images = skiio.imread(
                    ImageCollection.image_folder + os.sep + ImageCollection.image_list[indexes[num_images]])

            if mode =='HSV':
                images = skic.rgb2hsv(images)
            elif mode =='Lab':
                # L [0,100],a,b [-127,127]
                images = skic.rgb2lab(images)
                images = self.rescaleHistLab(images, n_bins)

            if mode !='Lab' : # do we want to do this for rgb ?
                # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
                images = np.round(images/np.max(images) * (n_bins - 1)).astype('int32')  # HSV has all values between 0 and 100
            else :
                images=images.astype('int32')
            #255x255x3

            tracker.compute_for_image(images,num_images)
            '''for i in range (3):
                current_r=images[:,:,i].reshape((256*256)) # 256x256 array
                for k in store[i]  :
                    if k not in self.raw_data:
                        store[i][k].append(self.s_to_fun[k](current_r))

            for k in self.vector_data :
                store[k].append(self.s_to_fun[k](images))'''


        tracker.compute_mean()
        '''for i in range(3):
            # revoir data et raw data
            store[i]['data']=store[i].copy()
            #del store[i]['data']['data']
            for k in store[i]:
                if k not in self.raw_data:
                    store[i][k]=np.round(np.mean(store[i][k]),1)'''

        return tracker


    def getDatasetTable(self,tracker,current_mode='RGB',n_bins=256):

        fig, ax = plt.subplots(figsize=(10,5 ))
        # hide axes
        fig.patch.set_visible(False)

        watch=None
        data=None
        #----- start for loop here
        for c_dataset in self.all_classes :
            result=self.getStat(self.enc_classes[c_dataset],current_mode,n_bins)
            data[c_dataset]=[result[i][watch] for i in range(3)]+['']

        ax.axis('off')
        ax.axis('tight')
        df = pd.DataFrame(data, index=self.enc_repr[current_mode] +['Color'])
        table=ax.table(cellText=df.values, rowLabels=df.index,colLabels=df.columns, loc='center',
                 cellLoc='center')
        table.scale(2, 2)
        for i,c_class in enumerate(self.all_classes):
            current_color =[x/(n_bins-1) for x in data[c_class][:3]] #normalize between[0,1]

            if current_mode =='HSV':
                #current_color=matplotlib.colors.hsv_to_rgb(current_color)
                current_color=skic.hsv2rgb(current_color)
            elif current_mode=='Lab':
                current_color[0]*=100
                for j in range(1,3):
                    current_color[j]=(current_color[j]-0.5)*220
                current_color=skic.lab2rgb(current_color)
            if watch in self.watch_var[:2]:
                table[(4, i)].set_facecolor(current_color)
        ax.set_title(f"{watch}")

        #-----------

        fig.tight_layout()
        fig.suptitle(f"Dataset table infor for {n_bins} bins")
        #plt.show()

    def getDatasetScatterGraph(self,current_mode='RGB',n_bins=256,watch=None):

        colors=['blue','green','black']
        colors={k:v for k,v in zip(self.all_classes,colors) }
        fig, ax = plt.subplots(1,3)
        fig_3d = plt.figure(figsize=(10, 10))
        ax_3d = fig_3d.add_subplot(projection='3d')
        # ----- start for loop here
        for c_dataset in self.all_classes:
            result = self.getStat(self.enc_classes[c_dataset], current_mode, n_bins)

            if watch == 'max vector':
                data = result[watch]
                R = [data[j][1][0] for j in range(len(data))]
                G = [data[j][1][1] for j in range(len(data))]
                B = [data[j][1][2] for j in range(len(data))]
                values_axes = {0:R,1:G,2:B}
                ax_3d.scatter(values_axes[0], values_axes[1], values_axes[2], alpha=0.4,
                              color=colors[c_dataset], marker='o', label=c_dataset)

                for i in range(3) :
                    ax[i].scatter(values_axes[i], values_axes[(i+1)%3], alpha=0.4,color=colors[c_dataset],marker='.')
                    data_ellipse=np.array([values_axes[i],values_axes[(i+1)%3]]).T
                    viewEllipse(data = data_ellipse,ax = ax[i], facecolor = colors[c_dataset],scale= 1,alpha=0.25)

            else:
                data = [result[i]['data'] for i in range(3)]

                ax_3d.scatter(data[0][watch],data[1][watch],data[2][watch], alpha=0.4,
                              color=colors[c_dataset],marker='o',label=c_dataset)

                for i in range(3) :
                    ax[i].scatter(data[i][watch], data[(i+1)%3][watch], alpha=0.4,color=colors[c_dataset],marker='.')
                    data_ellipse=np.array([data[i][watch],data[(i+1)%3][watch]]).T
                    viewEllipse(data = data_ellipse,ax = ax[i], facecolor = colors[c_dataset],scale= 1,alpha=0.25)

        ax_3d.set_xlabel(self.enc_repr[current_mode][0])
        ax_3d.set_ylabel(self.enc_repr[current_mode][1])
        ax_3d.set_zlabel(self.enc_repr[current_mode][2])
        ax_3d.legend()

        for i in range(3):
            labelx = self.enc_repr[current_mode][i]
            labely = self.enc_repr[current_mode][(i+1)%3]
            label=f"{current_mode[(i+1)%3]}=fct({current_mode[i]})"
            ax[i].set_title(label)
            ax[i].set(xlabel=labelx, ylabel=labely)
            ax[i].axes.set_aspect('equal')

        fig.suptitle(f"{current_mode} using {n_bins} watching {watch}")
        fig.tight_layout()
        fig_3d.suptitle(f"{current_mode} using {n_bins} watching {watch}")
