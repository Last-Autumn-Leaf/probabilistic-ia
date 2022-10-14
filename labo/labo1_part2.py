

import matplotlib.pyplot as plt
from TroisClasses import TroisClasses
import helpers.analysis as an
import numpy as np


##########################################
getCorr = lambda classe: np.corrcoef(classe[:,0],classe[:,1])[0][1]

def main():
    classe1=np.array([[-1,0] ,[1,0] ])
    classe2=np.array([[0,1] ,[0,-1]])
    classes=[classe1,classe2]
    extent=an.Extent(xmin=-2, xmax=2, ymin=-2, ymax=2)
    # Statistiques
    m1, cov1, valpr1, vectprop1 = an.calcModeleGaussien(classe1, '\nClasse 1')
    m2, cov2, valpr2, vectprop2 = an.calcModeleGaussien(classe2, '\nClasse 2')


    print("-------------------")
    Fx=lambda x :-1/(1+np.exp(abs(x)/2))
    Fy=lambda x :1/(1+np.exp(-x/2))
    F=lambda  x: np.stack( [Fx(x[:,0]),Fy(x[:,1])],axis=1)
    #print(classe1)

    # Données d'origine et données décorrélées
    an.view_classes(classes, extent)
    an.view_classes(an.decorrelate_withF(classes,F ),
                    an.Extent(array=np.matmul(extent.get_array(), vectprop1)))

    plt.show()


#####################################
if __name__ == '__main__':
    main()
