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

(sec:cnn)=
# Traiter des images : les réseaux convolutifs

```{code-cell}
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from scipy.signal import convolve2d
from notebook_utils import prepare_notebook_graphics, BLEU, ORANGE, VERT, GRIS
prepare_notebook_graphics()
```

Les perceptrons multicouches du [chapitre précédent](sec:mlp) fonctionnent bien sur des données **tabulaires**, où chaque individu est décrit par une liste de variables.
Pour les **images**, on utilise une autre famille de modèles, bien plus efficace : les **réseaux de neurones convolutifs** (_Convolutional Neural Networks_, CNN, ou _ConvNets_).
C'est grâce à eux que les ordinateurs ont appris à reconnaître des objets, des visages ou des chiffres manuscrits.

## Une image, c'est un tableau de nombres

Pour un ordinateur, une image est une grille de petits carrés appelés **pixels**, et chaque pixel est décrit par un ou plusieurs nombres qui indiquent son intensité lumineuse.

Voici par exemple un chiffre manuscrit en noir et blanc, de très petite taille (8 pixels sur 8) : à gauche tel qu'on le voit, à droite tel que l'ordinateur le « voit ».

```{code-cell}
:tags: [hide-input]

from sklearn.datasets import load_digits

chiffre = load_digits().images[0].astype(int)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].imshow(chiffre, cmap="gray_r")
axes[0].set_title("L'image")
axes[1].imshow(chiffre, cmap="gray_r", alpha=.15)
for i in range(8):
    for j in range(8):
        axes[1].text(j, i, chiffre[i, j], ha="center", va="center", fontsize=12)
axes[1].set_title("Les nombres stockés")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
```

Chaque nombre va de 0 (blanc) à 16 (noir).
Une image en noir et blanc est donc simplement un tableau de nombres à deux dimensions (hauteur × largeur).

Une image **en couleur** est composée de trois tableaux de ce type, appelés **canaux** : un pour l'intensité de rouge, un pour le vert et un pour le bleu (on parle d'image RVB, ou _RGB_ en anglais).

```{code-cell}
:tags: [hide-input]

chat = mpimg.imread("data/cat.jpg")

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.ravel()
axes[0].imshow(chat)
axes[0].set_title(f"Image couleur\n{chat.shape[0]} × {chat.shape[1]} pixels")
for c, (ax, nom) in enumerate(zip(axes[1:], ["rouge", "vert", "bleu"])):
    canal = np.zeros_like(chat)
    canal[:, :, c] = chat[:, :, c]
    ax.imshow(canal)
    ax.set_title(f"Canal {nom}")
for ax in axes:
    ax.axis("off")
```

Cette photo de chat est décrite par $360 \times 540 \times 3 = 583\,200$ nombres.

## Pourquoi pas un MLP ?

On pourrait être tenté de mettre tous les pixels d'une image bout à bout, et de donner cette longue liste de nombres à un MLP.
Cela pose deux problèmes.

**Trop de paramètres.** Avec notre photo de chat, la couche d'entrée aurait 583 200 neurones.
Avec une couche cachée de seulement 100 neurones, il faudrait déjà plus de **58 millions** de poids pour la première couche.

**On perd la notion de voisinage.** Dans une image, ce qui compte, ce sont les motifs formés par des pixels **voisins** : un contour, un coin, une texture, un œil…
En mettant les pixels bout à bout, le MLP perd l'information de qui est à côté de qui.
De plus, un même motif (une oreille de chat, par exemple) peut apparaître n'importe où dans l'image : un MLP devrait apprendre à le reconnaître séparément à chaque position possible.

Les réseaux convolutifs résolvent ces deux problèmes grâce à une opération appelée **convolution**.

## La convolution : une loupe qui glisse sur l'image

L'idée de la convolution est de faire glisser une petite fenêtre, appelée **filtre** (ou **noyau**, _kernel_ en anglais), sur toute l'image.
Un filtre est un petit tableau de nombres, par exemple de taille 3 × 3.

À chaque position :

1. on superpose le filtre à un petit morceau de l'image ;
2. on multiplie chaque nombre du filtre par le pixel situé en dessous ;
3. on additionne tous ces produits : cela donne **un** nombre, qui devient un pixel de l'image de sortie.

C'est exactement le calcul d'un neurone (une somme pondérée !), sauf que ce neurone ne regarde qu'un petit morceau de l'image à la fois, et que **le même filtre est réutilisé à toutes les positions**.

L'animation ci-dessous montre ce processus : l'image d'entrée est en bleu, le filtre 3 × 3 qui glisse est représenté en gris foncé, et l'image produite est en vert.

```{image} img/no_padding_no_strides.gif
:width: 250px
:align: center
```

<p style="text-align: center; font-size: small;">Source : V. Dumoulin, F. Visin, <a href="https://github.com/vdumoulin/conv_arithmetic"><em>A guide to convolution arithmetic for deep learning</em></a>.</p>

L'image produite s'appelle une **carte d'activation** (_feature map_).
Elle indique, pour chaque position de l'image, à quel point le motif « recherché » par le filtre y est présent.

### Des filtres qui détectent des motifs

Selon les nombres qu'il contient, un filtre va mettre en évidence des choses différentes.
Voici deux exemples appliqués à notre photo de chat (convertie en niveaux de gris) :

```{code-cell}
:tags: [hide-input]

gris = chat.mean(axis=2)

filtres = {
    "Filtre « flou »\n(moyenne des voisins)": np.ones((9, 9)) / 81.,
    "Filtre « contours verticaux »": np.array([[-1, 0, 1],
                                               [-2, 0, 2],
                                               [-1, 0, 1]]),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(gris, cmap="gray")
axes[0].set_title("Image d'origine")
for ax, (nom, f) in zip(axes[1:], filtres.items()):
    sortie = convolve2d(gris, f, mode="same", boundary="symm")
    if f.min() < 0:
        sortie = np.abs(sortie)
        ax.imshow(sortie, cmap="gray_r", vmax=np.percentile(sortie, 99))
    else:
        ax.imshow(sortie, cmap="gray")
    ax.set_title(nom)
for ax in axes:
    ax.axis("off")
```

* Le premier filtre remplace chaque pixel par la moyenne de ses voisins : l'image devient **floue**.
* Le second filtre calcule une différence entre les pixels de gauche et ceux de droite : il réagit fortement là où l'intensité change brusquement de gauche à droite, c'est-à-dire sur les **contours verticaux** (ici en noir).

Voici les nombres qui composent le filtre « contours verticaux » :

| | | |
|:-:|:-:|:-:|
| −1 | 0 | 1 |
| −2 | 0 | 2 |
| −1 | 0 | 1 |

Les pixels de gauche sont comptés négativement, ceux de droite positivement : si la zone est uniforme, tout s'annule et le résultat est proche de 0.

Dans un réseau convolutif, **on ne choisit pas ces nombres à la main** : ce sont les **paramètres** du modèle, appris par descente de gradient, exactement comme les poids d'un MLP.
Le réseau découvre ainsi lui-même les motifs utiles pour la tâche qu'on lui demande.

```{admonition} Pour les curieuses et les curieux
:class: dropdown, note

Pour une image en niveaux de gris $x$ et un filtre $f$ de taille $3 \times 3$ (dont on numérote les cases de $-1$ à $1$ dans chaque direction), le pixel $(i, j)$ de la carte d'activation vaut :

$$
    (x * f)(i, j) = \sum_{k=-1}^{1} \sum_{l=-1}^{1} f_{k, l} \, x_{i + k, j + l}
$$

Pour une image couleur, le filtre a lui aussi 3 canaux, et l'on somme en plus sur les canaux.
```

### Le _padding_

Vous avez peut-être remarqué dans l'animation précédente que l'image de sortie est un peu plus petite que l'image d'entrée : le filtre ne peut pas « déborder » de l'image.
Si l'on souhaite conserver la même taille, on peut ajouter une bordure de zéros autour de l'image avant d'appliquer le filtre : c'est le **_padding_** (rembourrage).

```{image} img/same_padding_no_strides.gif
:width: 250px
:align: center
```

<p style="text-align: center; font-size: small;">Avec <em>padding</em> (la bordure ajoutée est en pointillés) : l'image produite a la même taille que l'image d'entrée.</p>

## Les couches d'un réseau convolutif

### Couche de convolution

Une **couche de convolution** contient **plusieurs filtres**, qui sont appliqués en parallèle à la même image.
Chaque filtre produit sa propre carte d'activation, et ces cartes sont empilées les unes sur les autres : si une couche contient 6 filtres, sa sortie est une « image » à 6 canaux.
Comme pour les neurones d'un MLP, chaque filtre a aussi un biais, et on applique une fonction d'activation (en général ReLU) au résultat.

En `keras`, une couche de convolution s'écrit :

```python
from keras.layers import Conv2D

Conv2D(filters=6, kernel_size=5, padding="valid", activation="relu")
```

où `filters` est le nombre de filtres, `kernel_size` leur taille (ici 5 × 5), et `padding` vaut `"valid"` (pas de _padding_) ou `"same"` (avec _padding_, pour conserver la taille de l'image).

Le grand avantage par rapport à un MLP : le nombre de paramètres ne dépend **pas** de la taille de l'image, seulement du nombre et de la taille des filtres.

### Couche de _pooling_

Une **couche de _pooling_** (regroupement) sert à **réduire la taille** des cartes d'activation, pour résumer l'information et alléger les calculs des couches suivantes.

Le plus courant est le **_max pooling_** de taille 2 × 2 : on découpe l'image en carrés de 2 × 2 pixels qui ne se chevauchent pas, et l'on ne garde que la **valeur maximale** de chaque carré.
L'image obtenue est deux fois plus petite en largeur et en hauteur :

```{code-cell}
:tags: [remove-input]

entree = np.array([[1, 3, 2, 0],
                   [4, 6, 1, 1],
                   [0, 2, 9, 5],
                   [3, 1, 7, 8]])
sortie = entree.reshape(2, 2, 2, 2).max(axis=(1, 3))
couleurs = [[BLEU, ORANGE], [VERT, GRIS]]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                         gridspec_kw={"width_ratios": [2, 1]})
for ax, tab, n in [(axes[0], entree, 4), (axes[1], sortie, 2)]:
    for i in range(n):
        for j in range(n):
            bloc = (i // 2, j // 2) if n == 4 else (i, j)
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1,
                                       facecolor=couleurs[bloc[0]][bloc[1]],
                                       alpha=.35, edgecolor="black"))
            gras = n == 2 or entree[i, j] == sortie[i // 2, j // 2]
            ax.text(j + .5, n - .5 - i, tab[i, j], ha="center", va="center",
                    fontsize=18, fontweight="bold" if gras else "normal")
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_aspect("equal"); ax.axis("off")
axes[0].set_title("Carte d'activation (4 × 4)")
axes[1].set_title("Après max pooling 2 × 2")
fig.text(.6, .48, "→", fontsize=40, ha="center", va="center");
```

En `keras` :

```python
from keras.layers import MaxPool2D

MaxPool2D(pool_size=2)
```

Une couche de _pooling_ n'a **aucun paramètre** à apprendre : elle applique toujours la même opération.

### La « tête » de classification

Après plusieurs couches de convolution et de _pooling_, on obtient des cartes d'activation de petite taille, qui résument les motifs détectés dans l'image.
Pour décider à quelle classe appartient l'image, on termine le réseau par un petit **MLP** (on parle de **tête de classification**) :

1. une couche `Flatten` (« aplatir ») met toutes les valeurs des cartes d'activation bout à bout, pour en faire une simple liste de nombres ;
2. une ou plusieurs couches `Dense`, comme dans le chapitre précédent, produisent la prédiction finale (avec une activation softmax pour la classification à plusieurs classes).

## Un exemple complet : LeNet

L'un des premiers réseaux convolutifs, appelé **LeNet**, a été proposé en 1998 par Yann Le Cun et ses collègues pour reconnaître des chiffres manuscrits (il a notamment servi à lire automatiquement les montants sur des chèques).
Il enchaîne deux blocs « convolution + _pooling_ », puis une tête de classification :

```{image} img/lenet.png
:align: center
```

On lit ce schéma de gauche à droite : l'image d'entrée (32 × 32 pixels) traverse des couches de convolution, qui produisent des cartes d'activation de plus en plus nombreuses mais de plus en plus petites, avant d'être aplatie et passée à des couches denses, jusqu'aux 10 neurones de sortie (un par chiffre, de 0 à 9).

En `keras`, ce modèle s'écrit :

```{code-cell}
:tags: [remove-stderr]

from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense

model = Sequential([
    Input(shape=(32, 32, 1)),  # image 32x32, 1 canal
    Conv2D(filters=6, kernel_size=5, activation="relu"),
    MaxPool2D(pool_size=2),
    Conv2D(filters=16, kernel_size=5, activation="relu"),
    MaxPool2D(pool_size=2),
    Flatten(),                 # début de la tête
    Dense(units=120, activation="relu"),
    Dense(units=84, activation="relu"),
    Dense(units=10, activation="softmax")  # 10 classes
])
model.summary(line_length=65)
```

La colonne `Output Shape` permet de suivre la taille des données à chaque étape : 32 × 32 × 1 en entrée, puis 28 × 28 × 6 après la première convolution (6 filtres de taille 5 × 5, sans _padding_), puis 14 × 14 × 6 après le _pooling_, et ainsi de suite.

## Réutiliser un modèle déjà entraîné

Entraîner un grand réseau convolutif demande énormément d'images et de puissance de calcul.
Heureusement, de nombreux modèles ont déjà été entraînés sur d'immenses bases d'images (comme ImageNet, qui contient plus d'un million d'images réparties en 1 000 catégories) et sont disponibles librement : on parle de **modèles pré-entraînés**.

On peut les utiliser de deux façons :

* **tels quels**, si les catégories qu'ils connaissent correspondent à notre problème ;
* comme **point de départ** pour un nouveau problème : on garde les couches de convolution, qui ont appris à détecter des motifs visuels très généraux (contours, textures, formes…), et on remplace seulement la tête de classification par une nouvelle, adaptée à nos classes. C'est ce qu'on appelle l’**apprentissage par transfert** (_transfer learning_), ou le **_fine-tuning_** lorsqu'on ré-entraîne aussi, légèrement, les couches de convolution.

Cette approche permet d'obtenir de très bons résultats avec seulement quelques centaines d'images. Nous la mettrons en pratique en TP.

## Et pour d'autres types de données ?

L'idée de la convolution ne se limite pas aux images.
Pour des **séries temporelles** (des mesures prises au cours du temps, comme les données d'un accéléromètre), on utilise des convolutions à **une dimension** : le filtre glisse le long de l'axe du temps au lieu de glisser sur les deux dimensions d'une image.
Le principe est exactement le même : le réseau apprend des filtres qui détectent des motifs (un pic, une oscillation…) où qu'ils se trouvent dans la série.
En `keras`, on utilise alors les couches `Conv1D` et `MaxPool1D`.

## Récapitulatif

```{admonition} À retenir
:class: important

* Une image est un tableau de nombres (hauteur × largeur × canaux).
* Une **convolution** fait glisser un petit **filtre** sur l'image ; elle produit une **carte d'activation** qui indique où se trouve un motif donné.
* Les nombres des filtres sont des **paramètres appris**, et le même filtre est utilisé à toutes les positions de l'image : cela fait beaucoup moins de paramètres qu'un MLP.
* Un réseau convolutif enchaîne des couches de **convolution** (`Conv2D`) et de **_pooling_** (`MaxPool2D`), puis une **tête de classification** (`Flatten` puis `Dense`).
* On peut réutiliser des **modèles pré-entraînés** sur de grandes bases d'images et les adapter à son propre problème (**apprentissage par transfert**).
```

## Exercices

````{admonition} Exercice #1
Une image en couleur fait 64 pixels de haut et 64 pixels de large. Combien de nombres faut-il pour la décrire ?

```{admonition} Solution
:class: dropdown, tip

$64 \times 64 \times 3 = 12\,288$ nombres (3 canaux : rouge, vert, bleu).
```
````

````{admonition} Exercice #2
On applique un _max pooling_ 2 × 2 à la carte d'activation suivante. Qu'obtient-on ?

| | | | |
|---|---|---|---|
| 2 | 0 | 1 | 5 |
| 7 | 3 | 2 | 2 |
| 1 | 1 | 0 | 4 |
| 0 | 6 | 3 | 1 |

```{admonition} Solution
:class: dropdown, tip

On découpe en 4 carrés de 2 × 2 et on garde le maximum de chacun :

| | |
|---|---|
| 7 | 5 |
| 6 | 4 |
```
````

````{admonition} Exercice #3
Dans le résumé du modèle LeNet ci-dessus, la première couche `Conv2D` a 156 paramètres. Pouvez-vous expliquer ce nombre ?

_Indice : il y a 6 filtres de taille 5 × 5, l'image d'entrée a 1 canal, et chaque filtre a un biais._

```{admonition} Solution
:class: dropdown, tip

Chaque filtre contient $5 \times 5 \times 1 = 25$ poids, plus 1 biais, soit 26 paramètres.
Avec 6 filtres : $6 \times 26 = 156$ paramètres.

Remarquez que ce nombre ne dépend pas de la taille de l'image d'entrée !
```
````

`````{admonition} Exercice #4
Le jeu de données CIFAR-10 contient des images **en couleur** de 32 × 32 pixels, réparties en 10 classes (avion, voiture, oiseau, chat…).
Comment faudrait-il modifier le code du modèle LeNet pour l'utiliser sur ce jeu de données ?

````{admonition} Solution
:class: dropdown, tip

Seule la couche d'entrée change : les images ont maintenant 3 canaux au lieu de 1.

```python
Input(shape=(32, 32, 3))
```

La couche de sortie a déjà 10 neurones avec une activation softmax, ce qui convient pour 10 classes.
````
`````
