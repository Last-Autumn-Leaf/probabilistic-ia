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

    Requirement already satisfied: absl-py==1.2.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 1)) (1.2.0)
    Requirement already satisfied: anyio==3.6.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 2)) (3.6.1)
    Requirement already satisfied: appnope==0.1.3 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 3)) (0.1.3)
    Requirement already satisfied: argon2-cffi==21.3.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 4)) (21.3.0)
    Requirement already satisfied: argon2-cffi-bindings==21.2.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 5)) (21.2.0)
    Requirement already satisfied: asttokens==2.0.8 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 6)) (2.0.8)
    Requirement already satisfied: astunparse==1.6.3 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 7)) (1.6.3)
    Requirement already satisfied: attrs==22.1.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 8)) (22.1.0)
    Requirement already satisfied: backcall==0.2.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 9)) (0.2.0)
    Requirement already satisfied: beautifulsoup4==4.11.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 10)) (4.11.1)
    Requirement already satisfied: bleach==5.0.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 11)) (5.0.1)
    Requirement already satisfied: cachetools==5.2.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 12)) (5.2.0)
    Requirement already satisfied: certifi==2022.9.24 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 13)) (2022.9.24)
    Requirement already satisfied: cffi==1.15.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 14)) (1.15.1)
    Requirement already satisfied: charset-normalizer==2.1.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 15)) (2.1.1)
    Requirement already satisfied: cycler==0.11.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 16)) (0.11.0)
    Requirement already satisfied: debugpy==1.6.3 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 17)) (1.6.3)
    Requirement already satisfied: decorator==5.1.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 18)) (5.1.1)
    Requirement already satisfied: defusedxml==0.7.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 19)) (0.7.1)
    Requirement already satisfied: entrypoints==0.4 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 20)) (0.4)
    Requirement already satisfied: executing==1.1.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 21)) (1.1.1)
    Requirement already satisfied: fastjsonschema==2.16.2 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 22)) (2.16.2)
    Requirement already satisfied: flatbuffers==22.9.24 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 23)) (22.9.24)
    Requirement already satisfied: fonttools==4.37.4 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 24)) (4.37.4)
    Requirement already satisfied: gast==0.4.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 25)) (0.4.0)
    Requirement already satisfied: google-auth==2.12.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 26)) (2.12.0)
    Requirement already satisfied: google-auth-oauthlib==0.4.6 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 27)) (0.4.6)
    Requirement already satisfied: google-pasta==0.2.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 28)) (0.2.0)
    Requirement already satisfied: grpcio==1.49.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 29)) (1.49.1)
    Requirement already satisfied: h5py==3.7.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 30)) (3.7.0)
    Requirement already satisfied: idna==3.4 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 31)) (3.4)
    Requirement already satisfied: imageio==2.22.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 32)) (2.22.1)
    Requirement already satisfied: ipykernel==6.16.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 33)) (6.16.0)
    Requirement already satisfied: ipython==8.5.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 34)) (8.5.0)
    Requirement already satisfied: ipython-genutils==0.2.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 35)) (0.2.0)
    Requirement already satisfied: ipywidgets==8.0.2 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 36)) (8.0.2)
    Requirement already satisfied: jedi==0.18.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 37)) (0.18.1)
    Requirement already satisfied: Jinja2==3.1.2 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 38)) (3.1.2)
    Requirement already satisfied: joblib==1.2.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 39)) (1.2.0)
    Requirement already satisfied: jsonschema==4.16.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 40)) (4.16.0)
    Requirement already satisfied: jupyter==1.0.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 41)) (1.0.0)
    Requirement already satisfied: jupyter-console==6.4.4 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 42)) (6.4.4)
    Requirement already satisfied: jupyter-core==4.11.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 43)) (4.11.1)
    Requirement already satisfied: jupyter-server==1.21.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 44)) (1.21.0)
    Requirement already satisfied: jupyter_client==7.4.2 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 45)) (7.4.2)
    Requirement already satisfied: jupyterlab-pygments==0.2.2 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 46)) (0.2.2)
    Requirement already satisfied: jupyterlab-widgets==3.0.3 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 47)) (3.0.3)
    Requirement already satisfied: keras==2.10.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 48)) (2.10.0)
    Requirement already satisfied: Keras-Preprocessing==1.1.2 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 49)) (1.1.2)
    Requirement already satisfied: kiwisolver==1.4.4 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 50)) (1.4.4)
    Requirement already satisfied: libclang==14.0.6 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 51)) (14.0.6)
    Requirement already satisfied: Markdown==3.4.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 52)) (3.4.1)
    Requirement already satisfied: MarkupSafe==2.1.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 53)) (2.1.1)
    Requirement already satisfied: matplotlib==3.5.3 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 54)) (3.5.3)
    Requirement already satisfied: matplotlib-inline==0.1.6 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 55)) (0.1.6)
    Requirement already satisfied: mistune==2.0.4 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 56)) (2.0.4)
    Requirement already satisfied: mpmath==1.2.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 57)) (1.2.1)
    Requirement already satisfied: nbclassic==0.4.5 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 58)) (0.4.5)
    Requirement already satisfied: nbclient==0.7.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 59)) (0.7.0)
    Requirement already satisfied: nbconvert==7.2.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 60)) (7.2.1)
    Requirement already satisfied: nbformat==5.7.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 61)) (5.7.0)
    Requirement already satisfied: nest-asyncio==1.5.6 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 62)) (1.5.6)
    Requirement already satisfied: networkx==2.8.7 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 63)) (2.8.7)
    Requirement already satisfied: notebook==6.5.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 64)) (6.5.1)
    Requirement already satisfied: notebook-shim==0.1.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 65)) (0.1.0)
    Requirement already satisfied: numpy==1.23.4 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 66)) (1.23.4)
    Requirement already satisfied: oauthlib==3.2.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 67)) (3.2.1)
    Requirement already satisfied: opt-einsum==3.3.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 68)) (3.3.0)
    Requirement already satisfied: packaging==21.3 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 69)) (21.3)
    Requirement already satisfied: pandas==1.5.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 70)) (1.5.0)
    Requirement already satisfied: pandocfilters==1.5.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 71)) (1.5.0)
    Requirement already satisfied: parso==0.8.3 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 72)) (0.8.3)
    Requirement already satisfied: pexpect==4.8.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 73)) (4.8.0)
    Requirement already satisfied: pickleshare==0.7.5 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 74)) (0.7.5)
    Requirement already satisfied: Pillow==9.2.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 75)) (9.2.0)
    Requirement already satisfied: prometheus-client==0.15.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 76)) (0.15.0)
    Requirement already satisfied: prompt-toolkit==3.0.31 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 77)) (3.0.31)
    Requirement already satisfied: protobuf==3.19.6 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 78)) (3.19.6)
    Requirement already satisfied: psutil==5.9.2 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 79)) (5.9.2)
    Requirement already satisfied: ptyprocess==0.7.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 80)) (0.7.0)
    Requirement already satisfied: pure-eval==0.2.2 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 81)) (0.2.2)
    Requirement already satisfied: pyasn1==0.4.8 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 82)) (0.4.8)
    Requirement already satisfied: pyasn1-modules==0.2.8 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 83)) (0.2.8)
    Requirement already satisfied: pycparser==2.21 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 84)) (2.21)
    Requirement already satisfied: Pygments==2.13.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 85)) (2.13.0)
    Requirement already satisfied: pyparsing==3.0.9 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 86)) (3.0.9)
    Requirement already satisfied: pyrsistent==0.18.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 87)) (0.18.1)
    Requirement already satisfied: python-dateutil==2.8.2 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 88)) (2.8.2)
    Requirement already satisfied: pytz==2022.4 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 89)) (2022.4)
    Requirement already satisfied: PyWavelets==1.4.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 90)) (1.4.1)
    Requirement already satisfied: pyzmq==24.0.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 91)) (24.0.1)
    Requirement already satisfied: qtconsole==5.3.2 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 92)) (5.3.2)
    Requirement already satisfied: QtPy==2.2.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 93)) (2.2.1)
    Requirement already satisfied: requests==2.28.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 94)) (2.28.1)
    Requirement already satisfied: requests-oauthlib==1.3.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 95)) (1.3.1)
    Requirement already satisfied: rsa==4.9 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 96)) (4.9)
    Requirement already satisfied: scikit-image==0.19.3 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 97)) (0.19.3)
    Requirement already satisfied: scikit-learn==1.1.2 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 98)) (1.1.2)
    Requirement already satisfied: scipy==1.9.2 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 99)) (1.9.2)
    Requirement already satisfied: seaborn==0.12.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 100)) (0.12.1)
    Requirement already satisfied: Send2Trash==1.8.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 101)) (1.8.0)
    Requirement already satisfied: six==1.16.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 102)) (1.16.0)
    Requirement already satisfied: sniffio==1.3.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 103)) (1.3.0)
    Requirement already satisfied: soupsieve==2.3.2.post1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 104)) (2.3.2.post1)
    Requirement already satisfied: stack-data==0.5.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 105)) (0.5.1)
    Requirement already satisfied: sympy==1.11.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 106)) (1.11.1)
    Requirement already satisfied: tensorboard==2.10.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 107)) (2.10.1)
    Requirement already satisfied: tensorboard-data-server==0.6.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 108)) (0.6.1)
    Requirement already satisfied: tensorboard-plugin-wit==1.8.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 109)) (1.8.1)
    Requirement already satisfied: tensorflow==2.10.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 110)) (2.10.0)
    Requirement already satisfied: tensorflow-estimator==2.10.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 111)) (2.10.0)
    Requirement already satisfied: tensorflow-io-gcs-filesystem==0.27.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 112)) (0.27.0)
    Requirement already satisfied: termcolor==2.0.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 113)) (2.0.1)
    Requirement already satisfied: terminado==0.16.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 114)) (0.16.0)
    Requirement already satisfied: threadpoolctl==3.1.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 115)) (3.1.0)
    Requirement already satisfied: tifffile==2022.10.10 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 116)) (2022.10.10)
    Requirement already satisfied: tinycss2==1.1.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 117)) (1.1.1)
    Requirement already satisfied: tornado==6.2 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 118)) (6.2)
    Requirement already satisfied: traitlets==5.4.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 119)) (5.4.0)
    Requirement already satisfied: typing_extensions==4.4.0 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 120)) (4.4.0)
    Requirement already satisfied: urllib3==1.26.12 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 121)) (1.26.12)
    Requirement already satisfied: wcwidth==0.2.5 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 122)) (0.2.5)
    Requirement already satisfied: webencodings==0.5.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 123)) (0.5.1)
    Requirement already satisfied: websocket-client==1.4.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 124)) (1.4.1)
    Requirement already satisfied: Werkzeug==2.2.2 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 125)) (2.2.2)
    Requirement already satisfied: widgetsnbextension==4.0.3 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 126)) (4.0.3)
    Requirement already satisfied: wrapt==1.14.1 in ./venv/lib/python3.10/site-packages (from -r requirements.txt (line 127)) (1.14.1)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in ./venv/lib/python3.10/site-packages (from astunparse==1.6.3->-r requirements.txt (line 7)) (0.37.1)
    Requirement already satisfied: setuptools>=41.0.0 in ./venv/lib/python3.10/site-packages (from tensorboard==2.10.1->-r requirements.txt (line 107)) (60.2.0)
    Note: you may need to restart the kernel to use updated packages.


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


Ou depuis le note book


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
