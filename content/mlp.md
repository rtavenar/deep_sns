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

(sec:mlp)=
# Empiler des neurones : le perceptron multicouche

```{code-cell}
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from notebook_utils import prepare_notebook_graphics, draw_network, BLEU, ORANGE, VERT, GRIS
prepare_notebook_graphics()
```

Le perceptron du chapitre précédent ne contient qu'un seul neurone.
C'est un modèle simple à comprendre, mais aussi très limité.
Dans ce chapitre, nous allons voir comment assembler de nombreux neurones pour construire des modèles beaucoup plus riches : les **perceptrons multicouches**.

## Les limites d'un neurone seul

Un neurone seul calcule une somme pondérée de ses entrées.
Pour un problème de classification à deux classes, cela revient à séparer les deux classes **par une ligne droite** (ou, s'il y a plus de deux variables, par l'équivalent d'une droite en plus grande dimension).

Pour beaucoup de jeux de données, une droite ne suffit pas.
Dans l'exemple ci-dessous, les points bleus forment un disque entouré par les points orange : aucune droite ne permet de les séparer correctement.

```{code-cell}
:tags: [hide-input]

from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

X, y = make_circles(n_samples=300, noise=.1, factor=.4, random_state=0)
modeles = {
    "Un seul neurone": LogisticRegression(),
    "Réseau à une couche cachée": MLPClassifier(hidden_layer_sizes=(20,), max_iter=3000, random_state=0),
}

xx, yy = np.meshgrid(np.linspace(-1.6, 1.6, 300), np.linspace(-1.6, 1.6, 300))
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, (nom, modele) in zip(axes, modeles.items()):
    modele.fit(X, y)
    zz = modele.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-.5, .5, 1.5], colors=[ORANGE, BLEU], alpha=.2)
    ax.scatter(X[:, 0], X[:, 1], c=np.where(y == 1, BLEU, ORANGE), edgecolor="k", lw=.3)
    ax.set_title(f"{nom}\n(taux de bonnes réponses : {modele.score(X, y):.0%})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
```

Les zones colorées indiquent la classe prédite par chaque modèle en chaque point du plan.
À gauche, le neurone seul ne peut tracer qu'une frontière droite, et se trompe sur près d'un point sur deux.
À droite, un réseau composé de plusieurs neurones parvient à dessiner une frontière en forme de cercle, bien adaptée aux données.

## Empiler des couches de neurones

L'idée est la suivante : au lieu de relier directement les entrées à la sortie, on intercale entre les deux une **couche** de neurones.
Chacun de ces neurones reçoit toutes les entrées, calcule sa propre somme pondérée (avec ses propres poids), applique sa fonction d'activation, et transmet son résultat au neurone de sortie.

```{code-cell}
:tags: [remove-input]

draw_network([5, 7, 1],
             layer_names=["Entrée", "Couche\ncachée", "Sortie"]);
```

Sur ce schéma :

* la **couche d'entrée** (en bleu) ne fait aucun calcul : elle contient simplement les valeurs des variables explicatives d'un individu ;
* la **couche cachée** (en bleu clair) est composée de neurones qui calculent des résultats intermédiaires. On dit qu'elle est « cachée » car on n'observe pas directement ses valeurs : ni en entrée, ni en sortie ;
* la **couche de sortie** (en vert) produit la prédiction.

Chaque trait représente un poids, c'est-à-dire un paramètre du modèle.
On parle de couches **entièrement connectées** (ou **denses**) car chaque neurone d'une couche est relié à tous les neurones de la couche précédente.

Rien n'empêche d'empiler plusieurs couches cachées les unes après les autres :

```{code-cell}
:tags: [remove-input]

draw_network([5, 7, 7, 1],
             layer_names=["Entrée", "1re couche\ncachée", "2e couche\ncachée", "Sortie"],
             figsize=(10, 4.5));
```

Un tel modèle, avec une ou plusieurs couches cachées, s'appelle un **perceptron multicouche** (_Multi-Layer Perceptron_, ou **MLP**).
Lorsque le nombre de couches devient grand, on parle de réseau **profond**, d'où le terme _deep learning_.

### Pourquoi ça marche ?

On peut voir chaque couche cachée comme une étape de transformation des données : chaque neurone de la couche cachée calcule une nouvelle variable, à partir des variables d'entrée.
La couche de sortie fait alors sa prédiction à partir de ces nouvelles variables, construites automatiquement pendant l'apprentissage, plutôt qu'à partir des variables d'origine.

Il existe même un résultat mathématique, appelé **théorème d'approximation universelle**, qui affirme qu'avec **une seule** couche cachée contenant suffisamment de neurones, un MLP peut approcher d'aussi près qu'on le souhaite à peu près n'importe quelle relation entre les entrées et la sortie.

Ce résultat est rassurant, mais il ne dit pas tout :

* il ne dit pas **combien** de neurones il faut (cela peut être énorme) ;
* il ne garantit pas que la descente de gradient trouvera effectivement les bons paramètres.

En pratique, on observe qu'il est souvent plus efficace d'empiler plusieurs couches cachées de taille raisonnable plutôt qu'une seule couche gigantesque.

```{admonition} Pour les curieuses et les curieux
:class: dropdown, note

Pour un MLP à deux couches cachées, notées $h^{(1)}$ et $h^{(2)}$, les calculs s'écrivent, couche après couche :

\begin{align*}
  \text{pour chaque neurone } i \text{ de la 1re couche cachée :}\quad h^{(1)}_{i} &= \varphi \left( \sum_j w^{(0)}_{ij} x_{j} + b^{(0)}_{i} \right) \\
  \text{pour chaque neurone } i \text{ de la 2e couche cachée :}\quad h^{(2)}_{i} &= \varphi \left( \sum_j w^{(1)}_{ij} h^{(1)}_{j} + b^{(1)}_{i} \right) \\
  \text{sortie :}\quad \hat{y} &= \varphi_\text{out} \left( \sum_i w^{(2)}_{i} h^{(2)}_{i} + b^{(2)} \right)
\end{align*}

Chaque ligne est exactement le calcul d'un neurone vu au chapitre précédent : un MLP n'est rien d'autre que beaucoup de neurones simples organisés en couches.
```

## Choisir l'architecture d'un MLP

L’**architecture** d'un réseau, c'est sa forme : nombre de couches, nombre de neurones par couche, fonctions d'activation.
Certains de ces choix sont **imposés par le problème**, d'autres sont **libres**.

Reprenons le jeu de données Iris du [premier chapitre](sec:donnees) : on souhaite prédire l'espèce d'une fleur (3 espèces possibles) à partir de 4 mesures.

* La **couche d'entrée** a forcément autant de neurones qu'il y a de variables explicatives : ici, **4**.
* La **couche de sortie** dépend de ce que l'on veut prédire. Ici, on veut que le modèle donne, pour chaque espèce, la probabilité que la fleur appartienne à cette espèce : il faut donc **3** neurones de sortie, un par classe.
* En revanche, le **nombre de couches cachées** et le **nombre de neurones** dans chacune sont libres : ce sont des **hyperparamètres**, que l'on choisit nous-mêmes (souvent en essayant plusieurs possibilités).

Plus généralement, le nombre de neurones de sortie se choisit ainsi :

| Type de problème | Exemple | Neurones de sortie |
|---|---|---|
| Régression | Prédire un prix | 1 (un par quantité à prédire) |
| Classification binaire (2 classes) | Spam ou non ? | 1 (la probabilité de la classe « oui ») |
| Classification à $K$ classes | Espèce d'iris | $K$ (une probabilité par classe) |

## Les fonctions d'activation

Dans un MLP, chaque neurone applique une fonction d'activation au résultat de sa somme pondérée.
Ce choix est un autre hyperparamètre important.

### Pourquoi en a-t-on besoin ?

Si l'on n'utilisait aucune fonction d'activation (ou, ce qui revient au même, la fonction identité), empiler des couches ne servirait à rien : une somme pondérée de sommes pondérées reste une simple somme pondérée.
Un MLP sans activation, aussi profond soit-il, ne pourrait toujours tracer que des frontières droites, exactement comme un neurone seul.

On utilise donc des fonctions d'activation **non linéaires**, c'est-à-dire dont la courbe n'est pas une droite.
Les trois plus connues sont représentées ci-dessous :

```{code-cell}
:tags: [hide-input]

x = np.linspace(-4, 4, 200)
activations = {
    "tanh": np.tanh(x),
    "sigmoïde": 1. / (1. + np.exp(-x)),
    "ReLU": np.maximum(x, 0.),
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, (nom, valeurs) in zip(axes, activations.items()):
    ax.plot(x, valeurs, color=BLEU, lw=3)
    ax.axhline(0, color=GRIS, lw=.8)
    ax.axvline(0, color=GRIS, lw=.8)
    ax.set_ylim([-1.2, 4.2])
    ax.grid(alpha=.3)
    ax.set_title(nom)
```

* La **sigmoïde** écrase toute valeur entre 0 et 1 ;
* la **tangente hyperbolique** (tanh) a la même forme, mais entre −1 et 1 ;
* la **ReLU** (_Rectified Linear Unit_) remplace les valeurs négatives par 0 et laisse les valeurs positives inchangées.

C'est la ReLU qui est aujourd'hui le choix par défaut pour les couches cachées : elle est très simple à calculer et elle facilite l'apprentissage des réseaux profonds.

```{admonition} Pour les curieuses et les curieux
:class: dropdown, note

Les formules de ces trois fonctions sont :

\begin{align*}
    \text{sigmoïde}(x) &= \frac{1}{1 + e^{-x}} \\
    \text{tanh}(x) &= \frac{2}{1 + e^{-2x}} - 1 \\
    \text{ReLU}(x) &= \begin{cases}
                        x \text{ si } x > 0\\
                        0 \text{ sinon }
                      \end{cases}
\end{align*}
```

### Le cas particulier de la couche de sortie

Pour la couche de sortie, la fonction d'activation ne se choisit pas librement : elle doit produire des valeurs **cohérentes avec ce que l'on cherche à prédire**.

* En **régression**, on utilise souvent l'activation identité (appelée `"linear"` dans `keras`), qui permet de produire n'importe quel nombre. Si la quantité à prédire est forcément positive (un prix, par exemple), on peut utiliser la ReLU.
* En **classification binaire**, le neurone de sortie doit produire une probabilité, comprise entre 0 et 1 : on utilise la **sigmoïde**.
* En **classification à plusieurs classes**, on a un neurone par classe, et l'on veut que les sorties soient des probabilités : chacune entre 0 et 1, et leur somme égale à 1. On utilise pour cela la fonction **softmax**.

La fonction softmax transforme une liste de scores quelconques en une liste de probabilités, en conservant l'ordre : le score le plus élevé donne la probabilité la plus élevée.
Par exemple, pour un problème à 3 classes :

```{code-cell}
:tags: [hide-input]

scores = np.array([2.0, 1.0, -1.0])
probas = np.exp(scores) / np.sum(np.exp(scores))
pd.DataFrame({"Score avant softmax": scores,
              "Probabilité après softmax": probas.round(2)},
             index=["setosa", "versicolor", "virginica"])
```

Ici, le modèle prédit _setosa_ avec une probabilité de 70 % environ.

```{admonition} Pour les curieuses et les curieux
:class: dropdown, note

Si $o_1, \dots, o_K$ sont les scores des $K$ neurones de sortie (avant activation), la softmax calcule :

$$
  \text{softmax}(o_i) = \frac{e^{o_i}}{\sum_{j=1}^K e^{o_j}}
$$

L'exponentielle rend tous les scores positifs, et la division par la somme garantit que le total fait 1.
```

```{admonition} À retenir
:class: important

| Couche | Fonction d'activation habituelle |
|---|---|
| Couches cachées | ReLU |
| Sortie, régression | identité (`"linear"`), ou ReLU si la cible est positive |
| Sortie, classification binaire | sigmoïde |
| Sortie, classification à plusieurs classes | softmax |
```

## Déclarer un MLP en `keras`

Passons à la pratique.
La bibliothèque Python `keras` permet de déclarer un MLP en quelques lignes : il suffit d'empiler des couches dans un modèle `Sequential` (« séquentiel », car les couches s'enchaînent les unes après les autres).

Par exemple, pour déclarer un modèle composé de :

* une couche d'entrée à 10 variables ;
* une couche cachée de 20 neurones, avec activation ReLU ;
* une couche de sortie de 3 neurones, avec activation softmax (classification à 3 classes) ;

on écrit :

```{code-cell}
:tags: [remove-stderr]

from keras.models import Sequential
from keras.layers import Input, Dense

model = Sequential([
    Input(shape=(10, )),                  # entrée : 10 variables
    Dense(units=20, activation="relu"),   # couche cachée
    Dense(units=3, activation="softmax")  # couche de sortie
])

model.summary(line_length=65)
```

Quelques remarques :

* `Input(shape=(10, ))` indique la taille des entrées (le nombre de variables explicatives) ;
* `Dense` désigne une couche dense, c'est-à-dire entièrement connectée ; `units` est son nombre de neurones et `activation` sa fonction d'activation ;
* `model.summary()` affiche un résumé du modèle, avec le nombre de paramètres de chaque couche (colonne `Param #`).

Déclarer le modèle n'est que la première étape : il faudra ensuite l’**entraîner** sur des données, ce que nous ferons en travaux pratiques.

## Récapitulatif

```{admonition} À retenir
:class: important

* Un **perceptron multicouche** (MLP) est formé de neurones organisés en **couches** : une couche d'entrée, une ou plusieurs couches cachées, une couche de sortie.
* Les couches cachées permettent de représenter des relations bien plus complexes qu'avec un seul neurone.
* La taille des couches d'entrée et de sortie est **imposée par le problème** ; le nombre et la taille des couches cachées sont des **hyperparamètres**.
* Les fonctions d'activation **non linéaires** sont indispensables ; la ReLU est le choix par défaut pour les couches cachées.
* La fonction d'activation de la couche de sortie dépend du **type de problème**.
```

## Exercices

````{admonition} Exercice #1
Pouvez-vous expliquer le nombre de paramètres affiché par `model.summary()` pour le modèle déclaré plus haut ?

_Indice : chaque trait du schéma du réseau correspond à un poids, et chaque neurone (hors couche d'entrée) a en plus son propre biais._

```{admonition} Solution
:class: dropdown, tip

**Première couche dense.** Chacune des 10 entrées est reliée à chacun des 20 neurones cachés, ce qui fait $10 \times 20 = 200$ poids.
Chaque neurone caché a en plus son biais : $20$ paramètres supplémentaires.
Total : $200 + 20 = 220$ paramètres.

**Couche de sortie.** De la même manière : $20 \times 3 = 60$ poids, plus $3$ biais, soit $63$ paramètres.

**Total** : $220 + 63 = 283$ paramètres, ce qui correspond bien à ce qu'affiche `model.summary()`.
```
````

`````{admonition} Exercice #2
Déclarez, en `keras`, un MLP avec une couche cachée de 100 neurones (activation ReLU) pour le jeu de données Iris.

````{admonition} Solution
:class: dropdown, tip

Il y a 4 variables explicatives et 3 classes, d'où :

```python
model = Sequential([
    Input(shape=(4, )),
    Dense(units=100, activation="relu"),
    Dense(units=3, activation="softmax")
])
```
````
`````

`````{admonition} Exercice #3
Même question pour le jeu de données sur les logements de Boston présenté ci-dessous, où l'on cherche à prédire `PRICE` à partir des 6 autres colonnes.

````{admonition} Solution
:class: dropdown, tip

Il y a 6 variables explicatives, et il s'agit d'une régression avec une seule quantité à prédire, qui est un prix (donc positif) :

```python
model = Sequential([
    Input(shape=(6, )),
    Dense(units=100, activation="relu"),
    Dense(units=1, activation="relu")
])
```

L'activation `"linear"` pour la couche de sortie serait aussi un choix tout à fait raisonnable.
````
`````

```{code-cell}
:tags: [hide-input]

boston = pd.read_csv("data/boston.csv")[["RM", "CRIM", "INDUS", "NOX", "AGE", "TAX", "PRICE"]]
boston
```

````{admonition} Exercice #4
Combien de paramètres compte le modèle de l'exercice #2 ?

```{admonition} Solution
:class: dropdown, tip

Couche cachée : $4 \times 100 + 100 = 500$ paramètres.
Couche de sortie : $100 \times 3 + 3 = 303$ paramètres.
Total : $803$ paramètres.
```
````
