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

(sec:entrainement)=
# Entraîner un réseau : pertes et optimisation

```{code-cell}
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from notebook_utils import prepare_notebook_graphics, BLEU, ORANGE, VERT, GRIS
prepare_notebook_graphics()
```

Dans les chapitres précédents, nous avons vu comment **déclarer** un réseau de neurones.
Il reste à l’**entraîner**, c'est-à-dire à régler ses paramètres pour qu'il fasse de bonnes prédictions.
Pour cela, il faut faire deux choix, qui correspondent aux deux arguments principaux de `model.compile` dans `keras` :

* **la fonction de perte** (`loss`) : comment mesure-t-on les erreurs du modèle ?
* **l'optimiseur** (`optimizer`) : comment modifie-t-on les paramètres pour réduire ces erreurs ?

## Choisir la fonction de perte

Rappelons que la fonction de perte mesure l'erreur commise par le modèle : plus elle est petite, meilleur est le modèle.
Son choix dépend du **type de problème**.

### En régression : l'erreur quadratique moyenne

Nous l'avons déjà rencontrée dans le [premier chapitre](sec:donnees) : pour chaque exemple, on calcule l'écart entre la prédiction et la vraie valeur, on l'élève au carré, et on fait la moyenne sur tous les exemples.

Le carré a une conséquence importante : **les grosses erreurs coûtent très cher**.
Une erreur de 2 coûte 4, mais une erreur de 10 coûte 100.

```{code-cell}
:tags: [hide-input]

ecart = np.linspace(-4, 4, 100)
plt.plot(ecart, ecart ** 2, color=BLEU, lw=3)
plt.xlabel("Écart entre prédiction et vraie valeur")
plt.ylabel("Contribution à la perte");
```

### En classification : l'entropie croisée

En classification, le modèle ne prédit pas directement une classe : il prédit une **probabilité pour chaque classe** (grâce à la fonction softmax, ou à la sigmoïde pour deux classes).
Un bon modèle est un modèle qui donne une **forte probabilité à la bonne classe**.

L’**entropie croisée** (_cross-entropy_), aussi appelée perte logistique, mesure exactement cela.
Pour chaque exemple, on regarde la probabilité $p$ que le modèle a donnée à la bonne classe, et on calcule $-\log(p)$ :

```{code-cell}
:tags: [hide-input]

p = np.linspace(0.01, 1, 200)
plt.plot(p, -np.log(p), color=ORANGE, lw=3)
plt.xlabel("Probabilité donnée à la bonne classe")
plt.ylabel("Contribution à la perte");
```

* si le modèle donne une probabilité proche de 1 à la bonne classe, la perte est proche de 0 : très bien ;
* si le modèle hésite (probabilité de 0,5), la perte est moyenne ;
* si le modèle se trompe **en étant sûr de lui** (probabilité proche de 0 pour la bonne classe), la perte devient très grande.

Prenons un exemple sur le jeu de données Iris (3 espèces), pour une fleur dont la vraie espèce est _setosa_ :

| Modèle | Proba. _setosa_ | Proba. _versicolor_ | Proba. _virginica_ | Perte $-\log(p_\text{setosa})$ |
|---|---|---|---|---|
| A (sûr et juste) | 0,90 | 0,05 | 0,05 | 0,11 |
| B (hésitant) | 0,50 | 0,30 | 0,20 | 0,69 |
| C (sûr et faux) | 0,05 | 0,90 | 0,05 | 3,00 |

Comme pour l'erreur quadratique moyenne, la perte sur un jeu de données est la **moyenne** des pertes sur tous les exemples.

```{admonition} Pour les curieuses et les curieux
:class: dropdown, note

Pourquoi le logarithme ? Si les exemples sont indépendants, la probabilité que le modèle donne **toutes** les bonnes réponses est le produit des probabilités $p_1 \times p_2 \times \dots \times p_n$.
Maximiser ce produit revient à maximiser son logarithme $\log(p_1) + \dots + \log(p_n)$, c'est-à-dire à minimiser $-\frac{1}{n}\sum_i \log(p_i)$ : c'est exactement l'entropie croisée.
On parle de **maximum de vraisemblance**.
```

### Récapitulatif : sortie, activation et perte vont ensemble

Le choix de la perte va de pair avec celui de la couche de sortie, vu au [chapitre précédent](sec:mlp) :

| Problème | Neurones de sortie | Activation de sortie | Perte (`loss` dans `keras`) |
|---|---|---|---|
| Régression | 1 | `"linear"` (ou `"relu"`) | `"mse"` |
| Classification binaire | 1 | `"sigmoid"` | `"binary_crossentropy"` |
| Classification à $K$ classes | $K$ | `"softmax"` | `"categorical_crossentropy"` |

## Optimiser : la descente de gradient stochastique

### Le problème de la descente de gradient « classique »

Au [chapitre 2](sec:perceptron), à chaque pas de descente de gradient, nous calculions la pente de la perte en utilisant **tous** les exemples du jeu de données.
Avec 500 quartiers de Boston, c'est rapide.
Mais un jeu de données réel peut contenir des millions d'exemples (images, textes…) : il faudrait parcourir toutes les données **pour faire un seul petit pas**.

### L'idée : estimer la pente sur un petit échantillon

La **descente de gradient stochastique** (_Stochastic Gradient Descent_, SGD) part d'une idée simple, la même que celle d'un sondage : pour estimer une moyenne, pas besoin d'interroger tout le monde, un échantillon suffit.

On découpe donc les données d'entraînement, au hasard, en petits paquets appelés **mini-batchs** (_mini-batches_), par exemple de 32 ou 100 exemples.
Puis, pour chaque mini-batch :

1. on calcule la pente de la perte **sur ce mini-batch seulement** ;
2. on fait un pas de descente de gradient avec cette pente.

Quand tous les mini-batchs ont été utilisés une fois, on dit qu'on a fait une **époque** (_epoch_) : un passage complet sur les données d'entraînement.
On mélange alors les données, on refait des mini-batchs, et on recommence pour l'époque suivante.

```{admonition} Exemple
Avec 60 000 images d'entraînement et des mini-batchs de 100 images, chaque époque contient $60\,000 / 100 = 600$ mini-batchs : on fait donc **600 pas** de descente de gradient par époque, au lieu d'un seul.
```

Dans `keras`, la taille des mini-batchs et le nombre d'époques se règlent au moment de l'entraînement :

```python
model.fit(x_train, y_train, batch_size=100, epochs=10)
```

### Des pas plus nombreux, mais plus bruités

Chaque pas est calculé à partir d'un échantillon : il n'est donc pas parfaitement orienté vers le minimum, il est un peu « bruité ».
Mais comme on fait beaucoup plus de pas, on progresse **beaucoup plus vite**.
La figure suivante compare, sur l'exemple des logements de Boston du chapitre 2, l'évolution du paramètre $w$ pendant 3 époques :

```{code-cell}
:tags: [hide-input]

boston = pd.read_csv("data/boston.csv")
pieces, prix = boston["RM"].to_numpy(), boston["PRICE"].to_numpy()

def pente(w, x, y):
    return np.mean(2 * (w * x - y) * x)

taux, n_epoques, taille_batch = 2e-3, 3, 32
rng = np.random.default_rng(0)

# descente de gradient « classique » : 1 pas par époque
w_gd = [0.]
for e in range(n_epoques):
    w_gd.append(w_gd[-1] - taux * pente(w_gd[-1], pieces, prix))

# descente de gradient stochastique : 1 pas par mini-batch
w_sgd, temps_sgd = [0.], [0.]
n = len(prix)
for e in range(n_epoques):
    ordre = rng.permutation(n)
    for debut in range(0, n, taille_batch):
        idx = ordre[debut:debut + taille_batch]
        w_sgd.append(w_sgd[-1] - taux * pente(w_sgd[-1], pieces[idx], prix[idx]))
        temps_sgd.append(e + min(debut + taille_batch, n) / n)

plt.plot(temps_sgd, w_sgd, color=ORANGE, lw=1.5, label="Stochastique (mini-batchs de 32)")
plt.plot(range(n_epoques + 1), w_gd, "o-", color=BLEU, lw=3, label="Classique (toutes les données)")
plt.axhline(3.65, color=GRIS, ls="--", label="Meilleure valeur de w")
plt.xlabel("Nombre d'époques")
plt.ylabel("Valeur de w")
plt.legend();
```

En 3 époques, la version stochastique a atteint la meilleure valeur de $w$ (en oscillant un peu autour), alors que la version classique n'a fait que 3 pas et en est encore loin.

```{admonition} Pour les curieuses et les curieux
:class: dropdown, note

Comment calcule-t-on la pente de la perte par rapport à **chacun** des milliers de paramètres d'un réseau ?
On utilise un algorithme appelé **rétropropagation du gradient** (_backpropagation_), qui applique la règle de dérivation des fonctions composées couche par couche, en partant de la sortie et en remontant vers l'entrée.
Bonne nouvelle : `keras` fait ce calcul automatiquement, vous n'avez jamais à l'écrire vous-même.
```

### Adam : une descente de gradient améliorée

De nombreuses variantes de la descente de gradient stochastique existent.
La plus utilisée aujourd'hui s'appelle **Adam**. Elle ajoute deux idées :

* **l'élan** (_momentum_) : comme une bille qui dévale une pente, on tient compte des pas précédents pour lisser la trajectoire et ne pas être trop perturbé par le bruit des mini-batchs ;
* **des pas adaptés à chaque paramètre** : les paramètres dont la pente varie beaucoup font des pas plus prudents, ceux dont la pente est faible et régulière font des pas relativement plus grands.

En pratique, Adam fonctionne bien dans la plupart des situations avec son réglage par défaut (taux d'apprentissage de 0,001) : c'est un bon premier choix.

Dans `keras`, l'optimiseur se choisit au moment de la compilation, soit par son nom (avec les réglages par défaut), soit en précisant le taux d'apprentissage :

```python
from keras.optimizers import SGD, Adam

# Adam avec ses réglages par défaut
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Adam en choisissant le taux d'apprentissage
model.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=0.001))

# Descente de gradient stochastique « simple »
model.compile(loss="categorical_crossentropy",
              optimizer=SGD(learning_rate=0.01))
```

## Préparer les données : la mise à l'échelle

Un dernier ingrédient, souvent oublié, est indispensable pour que l'optimisation se passe bien : **mettre toutes les variables d'entrée à la même échelle**.

Dans le jeu de données de Boston, par exemple, le nombre de pièces vaut environ 6, alors que le taux d'imposition (`TAX`) vaut plusieurs centaines.
Les poids associés à ces variables auront alors des ordres de grandeur très différents, et un même taux d'apprentissage sera trop grand pour certains et trop petit pour d'autres.

La solution la plus courante est la **standardisation** : pour chaque variable, on soustrait sa moyenne et on divise par son écart-type.
Toutes les variables sont alors « centrées » autour de 0, avec une dispersion comparable.

La figure suivante compare l'entraînement d'un même réseau (une couche cachée de 64 neurones, descente de gradient stochastique) sur les 13 variables des données de Boston, avec et sans standardisation.
Attention, l'axe vertical est en **échelle logarithmique** : chaque graduation correspond à une multiplication par 1 000.

```{code-cell}
:tags: [hide-input, remove-stderr]

from keras.models import Sequential
from keras.layers import Input, Dense
from keras.utils import set_random_seed

X_brut = boston.drop(columns=["PRICE"]).to_numpy()
X_std = (X_brut - X_brut.mean(axis=0)) / X_brut.std(axis=0)

for nom, X, couleur in [("Sans standardisation", X_brut, GRIS),
                        ("Avec standardisation", X_std, BLEU)]:
    set_random_seed(0)
    model = Sequential([
        Input(shape=(13, )),
        Dense(units=64, activation="relu"),
        Dense(units=1, activation="linear")
    ])
    model.compile(loss="mse", optimizer="sgd")
    h = model.fit(X, prix, epochs=50, batch_size=32, verbose=0).history
    plt.plot(range(1, 51), h["loss"], color=couleur, lw=2.5, label=nom)
plt.yscale("log")
plt.xlabel("Époque")
plt.ylabel("Perte (entraînement)")
plt.legend();
```

Sans standardisation, les pas de descente de gradient sont beaucoup trop grands pour les poids associés aux variables qui prennent de grandes valeurs : la perte **explose** dès la première époque et le modèle est inutilisable.
Avec des données standardisées, la perte diminue normalement.

```{admonition} Attention
:class: warning
La moyenne et l'écart-type utilisés pour standardiser doivent être calculés **sur les données d'entraînement uniquement**, puis appliqués tels quels aux données de test.
Sinon, des informations sur le jeu de test « fuient » dans l'entraînement.
```

## Récapitulatif

```{admonition} À retenir
:class: important

* Pour entraîner un réseau, on choisit une **fonction de perte** et un **optimiseur** (`model.compile`).
* Perte : **erreur quadratique moyenne** (`"mse"`) en régression, **entropie croisée** (`"categorical_crossentropy"` ou `"binary_crossentropy"`) en classification. L'entropie croisée pénalise fortement les erreurs commises avec assurance.
* La **descente de gradient stochastique** fait un pas par **mini-batch** : beaucoup plus de pas par **époque**, un peu bruités, mais une progression bien plus rapide.
* **Adam** est une variante de la descente de gradient stochastique qui fonctionne bien dans la plupart des cas.
* Il faut **mettre les données à l'échelle** (standardisation) avant l'entraînement.
```

## Exercices

````{admonition} Exercice #1
Pour chacun des problèmes suivants, quelle fonction de perte utiliseriez-vous dans `keras` ?

1. Prédire la température de demain.
2. Prédire si un client va résilier son abonnement.
3. Reconnaître un chiffre manuscrit (de 0 à 9).

```{admonition} Solution
:class: dropdown, tip

1. Régression : `"mse"`.
2. Classification binaire : `"binary_crossentropy"` (avec un neurone de sortie et une activation sigmoïde).
3. Classification à 10 classes : `"categorical_crossentropy"` (avec 10 neurones de sortie et une activation softmax).
```
````

````{admonition} Exercice #2
Un modèle de classification donne une probabilité de 0,8 à la bonne classe pour un premier exemple, et de 0,1 pour un second exemple.
Lequel de ces deux exemples contribue le plus à l'entropie croisée ?

```{admonition} Solution
:class: dropdown, tip

Le second : $-\log(0{,}1) \approx 2{,}30$, contre $-\log(0{,}8) \approx 0{,}22$ pour le premier.
Le modèle se trompe nettement sur le second exemple, et la perte le pénalise fortement.
```
````

````{admonition} Exercice #3
On entraîne un modèle sur 1 000 exemples, avec `batch_size=50` et `epochs=20`.
Combien de pas de descente de gradient sont effectués au total ?

```{admonition} Solution
:class: dropdown, tip

Chaque époque contient $1\,000 / 50 = 20$ mini-batchs, donc 20 pas.
Sur 20 époques : $20 \times 20 = 400$ pas au total.
```
````
