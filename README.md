# Projet IA Probabiliste

## Description du Projet
Ceci est le dépôt du projet d'APP 2 - IA Probabiliste.

Le but du projet est d'implémenter trois méthodes de classifications d'images sans utiliser la position des pixels en entrées. Les trois classes sont les côtes, les rues et les fôrets. 

Les trois classificateurs implémentés sont :
 - **Bayes**
 - **KNN**
 - **Réseau de neurones** 
 
Dans le futur nous espérons pouvoir implémenter d'autres classificateur ainsi que d'autres prétraitement permettant de meilleurs résultats.

## Installer le projet
Fonctionne avec la version 3.10.6 de Python.

Pour installer les librairies du projet lancer la commande suivante:


```python
pip install -r requirements.txt
```



## Lancer le projet
Pour lancer une classification il suffit de lancer le main.py avec un des arguments suivant :
- RNN
- BAYES
- KNN

### Exemple:


```python
!python main.py bayes
```

    2022-11-01 23:49:37.062748: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    Watching  4 dims
    Pre processing of all the data finished in  0:00:28.234953
    Taux de classification moyen sur l'ensemble des classes, Bayes risque #1: 85.71428571428572%
    Figure(640x480)


Ou depuis le notebook


```python
%matplotlib notebook
%run main.py bayes
```

    2022-11-01 23:50:19.643654: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


    Watching  4 dims
    Pre processing of all the data finished in  0:00:27.498935
    Taux de classification moyen sur l'ensemble des classes, Bayes risque #1: 85.71428571428572%



    <IPython.core.display.Javascript object>



<div id='65403600-b51b-4e16-9eba-dcbcd8446de2'></div>



    <IPython.core.display.Javascript object>



<div id='53d22542-7b89-44a0-b4a6-d080a80d38c2'></div>


# Collaborateurs
- [Carl-André GASSETTE](https://github.com/TheShyDev-Yoogo)
- [Arnaud Filleul](https://github.com/aRnaZ91)
- [Benjamin langlois](https://github.com/Benjaminlanglois)
