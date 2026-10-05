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

(sec:donnees)=
# Apprendre à partir de données

```{code-cell}
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from notebook_utils import prepare_notebook_graphics
prepare_notebook_graphics()
```

Avant de parler de neurones, prenons le temps de fixer le vocabulaire.
Le _deep learning_ est une branche de l’**apprentissage automatique** (_machine learning_) : l'idée générale est d'écrire des programmes qui ne suivent pas des règles écrites à la main, mais qui **apprennent ces règles à partir d'exemples**.

## Un jeu de données, c'est un tableau

Partons d'un exemple concret : un jeu de données classique qui décrit des quartiers de la région de Boston (États-Unis) dans les années 1970.
Voici quelques-unes de ses colonnes :

```{code-cell}
:tags: [hide-input]

boston = pd.read_csv("data/boston.csv")[["RM", "AGE", "CRIM", "PRICE"]]
boston
```

Dans ce tableau :

* chaque **ligne** décrit un quartier : on parle d’**individu**, d’**observation** ou d’**exemple** ;
* chaque **colonne** est une **variable** (on dit aussi **caractéristique**, _feature_ en anglais) :
  * `RM` est le nombre moyen de pièces par logement,
  * `AGE` est la proportion de logements construits avant 1940,
  * `CRIM` est le taux de criminalité,
  * `PRICE` est le prix médian des logements (en milliers de dollars).

Supposons que notre objectif soit de **prédire le prix** (`PRICE`) d'un quartier à partir des autres informations.
On appelle alors :

* `PRICE` la **variable cible** (_target_), c'est-à-dire ce que l'on cherche à prédire ;
* les autres colonnes les **variables explicatives** (ou simplement les **entrées** du modèle).

Par convention, on note souvent $x$ les entrées et $y$ la cible.

## Deux grands types de problèmes

Selon la nature de la cible, on distingue deux grandes familles de problèmes.

**La régression** : la cible est un **nombre** qui peut prendre une infinité de valeurs.
Prédire un prix, une température ou une durée sont des problèmes de régression.
C'est le cas de notre exemple sur les logements de Boston.

**La classification** : la cible est une **catégorie** parmi un nombre fini de possibilités.
Prédire si un courriel est un spam ou non, reconnaître l'animal présent sur une photo, ou encore deviner l'espèce d'une fleur sont des problèmes de classification.

Voici un autre jeu de données très classique, qui décrit des fleurs d'iris par la longueur et la largeur de leurs pétales et de leurs sépales :

```{code-cell}
:tags: [hide-input]

iris = pd.read_csv("data/iris.csv", index_col=0)
iris["target"] = iris["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
iris
```

Ici, la cible (`target`) est l'espèce de la fleur, qui peut prendre seulement trois valeurs : _setosa_, _versicolor_ ou _virginica_.
Il s'agit donc d'un problème de **classification à 3 classes**.

```{code-cell}
:tags: [hide-input]

sns.scatterplot(data=iris, x="petal length (cm)", y="petal width (cm)", hue="target")
plt.xlabel("Longueur du pétale (cm)")
plt.ylabel("Largeur du pétale (cm)")
plt.legend(title="Espèce");
```

On voit sur cette figure que les trois espèces occupent des zones assez différentes : un modèle devrait pouvoir apprendre à les distinguer à partir de ces mesures.

## Un modèle, c'est une fonction avec des boutons de réglage

Un **modèle** est une fonction qui prend en entrée les variables explicatives d'un individu et qui renvoie une **prédiction** de la cible, que l'on note $\hat{y}$ (« y chapeau »).

Ce qui rend le modèle intéressant, c'est qu'il possède des **paramètres** : on peut les voir comme des boutons de réglage.
Selon la position de ces boutons, le modèle fera des prédictions différentes.

**Apprendre**, pour un modèle, c'est trouver un bon réglage de ces boutons, c'est-à-dire des valeurs des paramètres pour lesquelles les prédictions sont proches des vraies valeurs de la cible, sur les exemples dont on dispose.

## Mesurer les erreurs : la fonction de perte

Pour savoir si un réglage est « bon », il faut pouvoir mesurer l'erreur commise par le modèle.
C'est le rôle de la **fonction de perte** (_loss function_), qu'on appelle aussi fonction de coût.

Pour un problème de régression, un choix très courant consiste à :

1. calculer, pour chaque exemple, l'écart entre la prédiction et la vraie valeur ;
2. élever cet écart au carré (pour que les erreurs positives et négatives ne se compensent pas, et pour pénaliser fortement les grosses erreurs) ;
3. faire la moyenne sur tous les exemples.

On obtient l’**erreur quadratique moyenne** (_Mean Squared Error_, MSE).

Prenons un exemple avec trois quartiers :

| Quartier | Vrai prix $y$ | Prix prédit $\hat{y}$ | Écart | Écart au carré |
|---|---|---|---|---|
| A | 24 | 26 | 2 | 4 |
| B | 21 | 20 | −1 | 1 |
| C | 35 | 29 | −6 | 36 |

La perte vaut ici $(4 + 1 + 36) / 3 \approx 13{,}7$.
Plus elle est petite, meilleur est le modèle : un modèle parfait aurait une perte nulle.

```{admonition} À retenir
:class: important
Apprendre = chercher les valeurs des paramètres qui rendent la **fonction de perte** la plus petite possible.
On appelle cette recherche l’**optimisation**, ou encore l’**entraînement** du modèle.
```

## Données d'entraînement et données de test

Un modèle qui obtient une perte très faible sur les exemples qu'il a vus pendant l'apprentissage n'est pas forcément un bon modèle : il a peut-être simplement « appris par cœur » ces exemples.
Ce qui nous intéresse vraiment, c'est sa capacité à faire de bonnes prédictions sur de **nouvelles** données.

C'est pourquoi on sépare toujours les données disponibles en deux (au moins) :

* un **jeu d'entraînement**, utilisé pour régler les paramètres ;
* un **jeu de test**, mis de côté et utilisé uniquement à la fin pour évaluer le modèle sur des exemples qu'il n'a jamais vus.

Nous reviendrons sur ce point plus tard dans le cours.

## Exercices

````{admonition} Exercice #1
Pour chacune des tâches suivantes, s'agit-il de régression ou de classification ?

1. Prédire le nombre de vélos loués demain dans une station de Rennes.
2. Prédire si un étudiant validera ou non son semestre.
3. Prédire la langue dans laquelle est écrit un texte.
4. Prédire la note (sur 20) obtenue à un examen.

```{admonition} Solution
:class: dropdown, tip

1. Régression (un nombre).
2. Classification à 2 classes (on parle de classification **binaire**).
3. Classification (autant de classes que de langues envisagées).
4. On le traite en général comme une régression : la note est un nombre, et prédire 14,5 quand la vraie note est 15 est une « petite » erreur, ce que la régression prend en compte et pas la classification.
```
````

````{admonition} Exercice #2
Dans le jeu de données Iris présenté plus haut, combien y a-t-il d'individus ? De variables explicatives ? Quelle est la variable cible ?

```{admonition} Solution
:class: dropdown, tip

Il y a 150 individus (150 lignes), 4 variables explicatives (longueur et largeur du sépale, longueur et largeur du pétale), et la cible est l'espèce de la fleur (`target`).
```
````
