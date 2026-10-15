---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(sec:surapprentissage)=
# Bien généraliser : le sur-apprentissage

```{code-cell}
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from notebook_utils import prepare_notebook_graphics, BLEU, CYAN, ORANGE, VERT, GRIS
prepare_notebook_graphics()
```

Un réseau de neurones avec beaucoup de paramètres peut représenter des relations très complexes.
C'est sa force… et aussi son principal danger : il peut finir par **apprendre par cœur** ses données d'entraînement, au lieu d'apprendre des règles générales.
Ce chapitre explique comment détecter ce phénomène et comment s'en protéger.

## Qu'est-ce que le sur-apprentissage ?

Imaginez un étudiant qui prépare un examen en apprenant par cœur les corrigés des annales.
Il obtiendra 20/20 sur les annales… mais sera perdu face à un sujet nouveau.
Il n'a pas compris le cours : il a mémorisé des réponses.

Un modèle peut faire exactement la même chose.
On dit qu'il fait du **sur-apprentissage** (_overfitting_) lorsqu'il est excellent sur ses données d'entraînement mais mauvais sur de nouvelles données.
À l'inverse, un modèle trop simple, qui n'arrive même pas à bien prédire sur ses données d'entraînement, fait du **sous-apprentissage** (_underfitting_).

La figure suivante illustre ces deux situations sur un petit problème de régression, avec des modèles de plus en plus flexibles :

```{code-cell}
:tags: [hide-input]

rng = np.random.default_rng(1)
x = np.sort(rng.uniform(0, 1, 15))
y = np.sin(2 * np.pi * x) + rng.normal(0, .25, x.size)
xx = np.linspace(0, 1, 300)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, degre, titre in zip(axes, [1, 3, 14],
                            ["Trop simple :\nsous-apprentissage", "Bien ajusté",
                             "Trop flexible :\nsur-apprentissage"]):
    coefs = np.polyfit(x, y, degre)
    ax.plot(xx, np.sin(2 * np.pi * xx), color=GRIS, ls="--", label="Vraie relation")
    ax.plot(xx, np.polyval(coefs, xx), color=ORANGE, lw=2.5, label="Modèle")
    ax.scatter(x, y, color=BLEU, zorder=3, label="Données d'entraînement")
    ax.set_ylim(-2, 2)
    ax.set_title(titre)
    ax.set_xticks([])
axes[0].legend(loc="lower left", fontsize=10);
```

* À gauche, le modèle (une droite) est trop simple pour suivre les données.
* Au milieu, le modèle capte la forme générale de la relation.
* À droite, le modèle passe exactement par tous les points d'entraînement… mais en faisant des détours absurdes entre eux : il a appris le **bruit** des données, pas la relation. Ses prédictions sur de nouveaux points seraient très mauvaises.

## Détecter le sur-apprentissage : le jeu de validation

Pour savoir si un modèle généralise bien, il faut le tester sur des données qu'il **n'a pas vues** pendant l'entraînement.
C'est le rôle du **jeu de validation** : une partie des données d'entraînement que l'on met de côté, et sur laquelle on mesure la perte à la fin de chaque époque.

Dans `keras`, l'argument `validation_split` de `fit` s'en charge : avec `validation_split=0.3`, les derniers 30 % des données d'entraînement servent de jeu de validation.

```{admonition} Validation ou test ?
Le **jeu de validation** sert à faire des choix pendant la construction du modèle (architecture, nombre d'époques…).
Le **jeu de test** n'est utilisé qu’**une seule fois**, à la toute fin, pour mesurer la performance du modèle retenu.
Si l'on utilisait le jeu de test pour faire des choix, il ne mesurerait plus une performance « sur des données nouvelles ».
```

On trace alors les **courbes d'apprentissage** : l'évolution de la perte d'entraînement et de la perte de validation au fil des époques.
Voici celles d'un réseau de 3 couches cachées de 256 neurones, entraîné sur un petit jeu de données de classification (200 exemples) :

```{code-cell}
:tags: [hide-input, remove-stderr]

from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from keras.utils import set_random_seed

X, y = make_moons(n_samples=200, noise=.35, random_state=0)

def cree_mlp():
    model = Sequential([
        Input(shape=(2, )),
        Dense(units=256, activation="relu"),
        Dense(units=256, activation="relu"),
        Dense(units=256, activation="relu"),
        Dense(units=1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

set_random_seed(0)
h = cree_mlp().fit(X, y, validation_split=0.3, epochs=200, batch_size=16, verbose=0).history

def trace_courbes(h, ax=None, titre=""):
    ax = ax or plt.gca()
    epoques = np.arange(1, len(h["loss"]) + 1)
    ax.plot(epoques, h["loss"], color=BLEU, lw=2, label="Entraînement")
    ax.plot(epoques, h["val_loss"], color=ORANGE, lw=2, label="Validation")
    meilleure = np.argmin(h["val_loss"]) + 1
    ax.axvline(meilleure, color=GRIS, ls="--")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Perte")
    ax.set_title(titre)
    ax.legend()
    return meilleure

meilleure_epoque = trace_courbes(h)
print(f"La perte de validation est la plus basse à l'époque {meilleure_epoque}.")
```

Ces courbes racontent une histoire typique :

* au début, les deux pertes diminuent ensemble : le modèle apprend des règles utiles ;
* puis la perte de validation atteint un minimum (trait pointillé) et **remonte**, alors que la perte d'entraînement continue de baisser : le modèle commence à apprendre par cœur. C'est le signe du sur-apprentissage.

```{admonition} À retenir
:class: important
Un **écart qui se creuse** entre la perte d'entraînement (qui baisse) et la perte de validation (qui stagne ou remonte) est le signe du **sur-apprentissage**.
```

## Se protéger du sur-apprentissage

Les techniques qui limitent le sur-apprentissage s'appellent des techniques de **régularisation**.
En voici deux.

### 1. Avoir plus de données (ou un modèle plus petit)

Plus il y a d'exemples, plus il est difficile de tous les apprendre par cœur.
Quand ce n'est pas possible, on peut aussi réduire la taille du modèle (moins de couches, moins de neurones) : il aura moins de capacité à mémoriser.

### 2. L'arrêt précoce (_early stopping_)

Puisque la perte de validation finit par remonter, l'idée la plus simple consiste à **arrêter l'entraînement au bon moment** : lorsque la perte de validation ne s'améliore plus.

En pratique, la perte de validation fluctue un peu d'une époque à l'autre : on attend donc quelques époques avant de décider qu'elle ne s'améliorera plus.
Ce nombre d'époques d'attente s'appelle la **patience**.
À l'arrêt, on récupère les paramètres de la meilleure époque.

Dans `keras`, cela se fait avec un **callback** (un outil qui intervient pendant l'entraînement), passé à `fit` :

```{code-cell}
:tags: [remove-stderr]

from keras.callbacks import EarlyStopping

arret_precoce = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
```

* `monitor="val_loss"` : on surveille la perte de validation ;
* `patience=20` : on s'arrête si elle ne s'est pas améliorée pendant 20 époques ;
* `restore_best_weights=True` : à l'arrêt, on revient aux paramètres de la meilleure époque.

```{code-cell}
:tags: [hide-input, remove-stderr]

set_random_seed(0)
h_es = cree_mlp().fit(X, y, validation_split=0.3, epochs=200, batch_size=16,
                      verbose=0, callbacks=[arret_precoce]).history
meilleure = trace_courbes(h_es)
plt.xlim(0, 200)
print(f"Entraînement arrêté après {len(h_es['loss'])} époques (au lieu de 200) ; "
      f"paramètres restaurés à l'époque {meilleure}.")
```

## Récapitulatif

```{admonition} À retenir
:class: important

* **Sur-apprentissage** : le modèle apprend par cœur ses données d'entraînement et généralise mal à de nouvelles données.
* On le détecte grâce à un **jeu de validation** : la perte de validation stagne ou remonte alors que la perte d'entraînement continue de baisser.
* Pour le limiter : plus de données ou un modèle plus petit, l’**arrêt précoce** (`EarlyStopping`).
* Le **jeu de test** ne sert qu'une fois, à la fin, pour évaluer le modèle retenu.
```

## Exercices

````{admonition} Exercice #1
Pendant l'entraînement d'un modèle, vous observez qu'à partir de l'époque 15 la perte d'entraînement continue de diminuer alors que la perte de validation augmente.
Que se passe-t-il, et que pouvez-vous faire ?

```{admonition} Solution
:class: dropdown, tip

Le modèle sur-apprend à partir de l'époque 15 environ.
On peut arrêter l'entraînement plus tôt (arrêt précoce, avec `EarlyStopping` et `restore_best_weights=True`), réduire la taille du modèle ou, si possible, utiliser plus de données.
```
````

````{admonition} Exercice #2
Un modèle obtient une perte élevée **à la fois** sur les données d'entraînement et sur les données de validation. Est-ce du sur-apprentissage ?

```{admonition} Solution
:class: dropdown, tip

Non : le modèle n'arrive même pas à bien prédire sur ses données d'entraînement. C'est plutôt du **sous-apprentissage** : il faudrait un modèle plus flexible (plus de neurones, plus de couches), ou l'entraîner plus longtemps.
```
````

````{admonition} Exercice #3
Pourquoi ne faut-il pas utiliser le jeu de test pour choisir le nombre d'époques ?

```{admonition} Solution
:class: dropdown, tip

Parce que le jeu de test doit mesurer la performance sur des données qui n'ont servi à **aucun** choix.
Si on choisit le nombre d'époques en regardant le jeu de test, on « optimise » le modèle pour ce jeu de test, et la performance mesurée dessus devient trop optimiste.
C'est le rôle du jeu de validation.
```
````
