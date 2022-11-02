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

Il est ausis possible de visualiser les différentes métriques via :
- 1d
- 2d
- conf

### Exemple:


```python
!python main.py bayes
```


Ou depuis le note book


```python
%matplotlib notebook
%run main.py bayes
```


# Collaborateurs
- [Carl-André GASSETTE](https://github.com/TheShyDev-Yoogo)
- [Arnaud Filleul](https://github.com/aRnaZ91)
- [Benjamin langlois](https://github.com/Benjaminlanglois)
