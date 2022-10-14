"""
Départ du laboratoire
Classification de 3 classes avec toutes les méthodes couvertes par l'APP
APP2 S8 GIA
"""


import matplotlib.pyplot as plt
from TroisClasses import TroisClasses
import helpers.analysis as an
import numpy as np


##########################################
getCorr = lambda classe: np.corrcoef(classe[:,0],classe[:,1])[0][1]

def main():
    # TODO voir L1.E2, L2.E1, L2.E2, L2.E3
    # Statistiques
    m1, cov1, valpr1, vectprop1 = an.calcModeleGaussien(TroisClasses.C1, '\nClasse 1')
    m2, cov2, valpr2, vectprop2 = an.calcModeleGaussien(TroisClasses.C2, '\nClasse 2')
    m3, cov3, valpr3, vectprop3 = an.calcModeleGaussien(TroisClasses.C3, '\nClasse 3')

    #L1.E2.3
    var1,var2=np.var(TroisClasses.C1,axis=0)
    rho=getCorr(TroisClasses.C1)
    print(rho,var1,var2)

    print("-------------------")
    # L1.E2.4
    print(valpr1, vectprop1 )




    # Données d'origine et données décorrélées
    an.view_classes([TroisClasses.C1, TroisClasses.C2, TroisClasses.C3], TroisClasses.extent)
    an.view_classes(an.decorrelate([TroisClasses.C1, TroisClasses.C2, TroisClasses.C3], vectprop1),
                    an.Extent(array=np.matmul(TroisClasses.extent.get_array(), vectprop1)))
    # TODO JB: bug la projection de l'extent ne semble pas englober l'entièreté du graphique!?
    # TODO bug dans decorrelate ou dans la projection de l'extent??

    # exemple d'une densité de probabilité arbitraire pour 1 classe
    an.creer_hist2D(TroisClasses.C1, 'C1')

    # classification

    plt.show()


#####################################
if __name__ == '__main__':
    main()
