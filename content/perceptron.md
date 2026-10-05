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

(sec:perceptron)=
# Un premier neurone : le perceptron

```{code-cell}
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from notebook_utils import prepare_notebook_graphics, draw_neuron, BLEU, ORANGE, VERT, GRIS
prepare_notebook_graphics()
```

Dans ce chapitre, nous présentons le plus simple des réseaux de neurones : un réseau constitué d’**un seul neurone**, appelé **perceptron**.
Il nous servira à introduire deux idées que l'on retrouvera dans tous les modèles plus complexes : ce que calcule un neurone, et comment on règle ses paramètres.

## Ce que calcule un neurone

Un neurone artificiel reçoit plusieurs nombres en entrée (les variables explicatives d'un individu) et renvoie un seul nombre en sortie.
Il procède en deux temps :

1. il calcule une **somme pondérée** de ses entrées : chaque entrée est multipliée par un **poids**, on additionne le tout, et on ajoute un nombre supplémentaire appelé **biais** ;
2. il applique à ce résultat une **fonction d'activation**, une transformation choisie à l'avance.

```{code-cell}
:tags: [remove-input]

draw_neuron();
```

Les poids $w_1, w_2, w_3$ et le biais $b$ sont les **paramètres** du neurone : ce sont eux qui seront réglés pendant l'apprentissage.

Prenons un exemple chiffré, avec trois entrées :

| | Entrée 1 | Entrée 2 | Entrée 3 |
|---|---|---|---|
| Valeur de l'entrée $x$ | 2 | 1 | 3 |
| Poids $w$ | 0,5 | −1 | 2 |
| Produit $w \times x$ | 1 | −1 | 6 |

Avec un biais $b = 1$, la somme pondérée vaut $1 - 1 + 6 + 1 = 7$.
Le neurone applique ensuite sa fonction d'activation à ce nombre 7 pour produire sa sortie.
Si la fonction d'activation est l'identité (« ne rien changer »), la sortie vaut simplement 7.
Nous verrons d'autres fonctions d'activation [au chapitre suivant](sec:mlp).

```{admonition} Pour les curieuses et les curieux
:class: dropdown, note

Avec $d$ entrées $x_1, \dots, x_d$, la sortie $a$ d'un neurone s'écrit :

$$
    a = \varphi\left( \sum_{j=1}^{d} w_j x_j + b \right)
$$

où $\varphi$ (la lettre grecque « phi ») désigne la fonction d'activation.
```

## Régler les paramètres : la descente de gradient

Reprenons les données sur les logements de Boston du [chapitre précédent](sec:donnees), et essayons de prédire le prix (`PRICE`) à partir d'une seule variable : le nombre moyen de pièces par logement (`RM`).

```{code-cell}
:tags: [hide-input]

boston = pd.read_csv("data/boston.csv")[["RM", "PRICE"]]
x = boston["RM"].to_numpy()
y = boston["PRICE"].to_numpy()

sns.scatterplot(data=boston, x="RM", y="PRICE", color=BLEU, alpha=.6)
plt.xlabel("Nombre moyen de pièces")
plt.ylabel("Prix médian (milliers de $)");
```

Sans surprise, plus les logements ont de pièces, plus ils sont chers.

Utilisons le modèle le plus simple possible : un neurone avec une seule entrée, pas de biais et pas de fonction d'activation.
Il n'a donc qu'un seul paramètre, le poids $w$, et sa prédiction est :

$$
    \text{prix prédit} = w \times \text{nombre de pièces}
$$

Graphiquement, ce modèle est une droite qui passe par l'origine, et $w$ règle sa pente.
Voici ce que l'on obtient pour quelques valeurs de $w$ :

```{code-cell}
:tags: [hide-input]

def mse(w, x, y):
    """Erreur quadratique moyenne du modèle « prix = w * pièces », pour une ou plusieurs valeurs de w."""
    w = np.atleast_1d(w)
    return np.mean((w[:, None] * x[None, :] - y[None, :]) ** 2, axis=1)

xx = np.linspace(3.3, 9, 10)
plt.scatter(x, y, color=BLEU, alpha=.3)
for w_val, color in zip([2, 3.6, 6], [ORANGE, VERT, GRIS]):
    plt.plot(xx, w_val * xx, color=color, lw=3,
             label=f"w = {w_val} (perte : {mse(w_val, x, y)[0]:.0f})")
plt.xlabel("Nombre moyen de pièces")
plt.ylabel("Prix médian (milliers de $)")
plt.legend();
```

Avec $w = 2$, les prix prédits sont trop bas ; avec $w = 6$, ils sont trop hauts.
La valeur $w = 3{,}6$ semble bien meilleure, ce que confirme la perte (l'erreur quadratique moyenne) indiquée dans la légende.

### Le paysage de la perte

Puisque notre modèle n'a qu'un seul paramètre, on peut tracer la valeur de la perte pour de nombreuses valeurs de $w$ :

```{code-cell}
:tags: [hide-input]

ww = np.linspace(-2, 10, num=200)
plt.plot(ww, mse(ww, x, y), color=ORANGE, lw=3)
plt.xlabel("Valeur du paramètre w")
plt.ylabel("Perte");
```

Le meilleur réglage correspond au point le plus bas de cette courbe, autour de $w \approx 3{,}6$.

Ici, on a trouvé ce minimum en essayant 200 valeurs de $w$ et en gardant la meilleure.
Mais cette méthode « par essais » est inutilisable en pratique : les réseaux de neurones ont des milliers, voire des milliards de paramètres, et le nombre de combinaisons à essayer serait astronomique.
Il nous faut une méthode plus astucieuse.

### Descendre la pente, pas à pas

Imaginez que vous êtes en randonnée en montagne, dans un épais brouillard, et que vous cherchez à rejoindre le fond de la vallée.
Vous ne voyez pas le paysage, mais vous sentez sous vos pieds dans quelle direction le sol descend.
Une stratégie raisonnable : faire un pas dans la direction qui descend, puis recommencer, encore et encore.

C'est exactement l'idée de la **descente de gradient** :

1. on part d'une valeur initiale du paramètre, choisie au hasard (ici, $w = 0$) ;
2. on calcule la **pente** de la courbe de perte à cet endroit (c'est ce qu'on appelle le **gradient**) ;
3. on déplace le paramètre d'un petit pas **dans le sens où la perte diminue** ;
4. on recommence à l'étape 2, jusqu'à ce que la perte ne diminue plus.

Le grand avantage de cette méthode, c'est qu'elle ne nécessite de connaître la pente qu'à l'endroit où l'on se trouve : pas besoin d'explorer tout le paysage.
Et elle fonctionne de la même manière lorsqu'il y a des millions de paramètres.

```{code-cell}
:tags: [hide-input]

def grad_mse(w, x, y):
    """Pente de la courbe de perte au point w."""
    return np.mean(2 * (w * x - y) * x)

def descente_de_gradient(w_init, taux, n_pas, x, y):
    w = [w_init]
    for t in range(n_pas):
        w.append(w[-1] - taux * grad_mse(w[-1], x, y))
    return np.array(w)

w_traj = descente_de_gradient(w_init=0., taux=5e-3, n_pas=10, x=x, y=y)

plt.plot(ww, mse(ww, x, y), color=ORANGE, lw=3, alpha=.5)
plt.plot(w_traj, mse(w_traj, x, y), "ko-")
plt.text(w_traj[0] + .2, mse(w_traj[0], x, y)[0], "départ")
plt.text(w_traj[-1] + .2, mse(w_traj[-1], x, y)[0] + 30, "après 10 pas")
plt.xlabel("Valeur du paramètre w")
plt.ylabel("Perte");
```

Chaque point noir correspond à une étape de l'algorithme.
On remarque que les premiers pas sont grands (la pente est forte, loin du minimum) et que les pas suivants sont de plus en plus petits à mesure que l'on approche du fond de la vallée, où la pente devient presque nulle.

### La taille des pas : le taux d'apprentissage

La taille des pas est réglée par un nombre appelé **taux d'apprentissage** (_learning rate_).
Ce n'est pas un paramètre appris par le modèle : c'est un réglage que l'on choisit nous-mêmes, à l'avance.
On appelle ce genre de réglage un **hyperparamètre**.

Son choix est délicat, comme le montre la figure suivante, où l'on effectue 10 pas avec trois taux d'apprentissage différents :

```{code-cell}
:tags: [hide-input]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, taux, titre in zip(axes,
                           [5e-4, 5e-3, 2.55e-2],
                           ["Trop petit", "Bien choisi", "Trop grand"]):
    w_traj = descente_de_gradient(w_init=0., taux=taux, n_pas=10, x=x, y=y)
    ax.plot(ww, mse(ww, x, y), color=ORANGE, lw=3, alpha=.5)
    ax.plot(w_traj, mse(w_traj, x, y), "ko-")
    ax.set_title(titre)
    ax.set_xlabel("Valeur du paramètre w")
axes[0].set_ylabel("Perte");
```

* Avec un taux **trop petit**, les pas sont minuscules : on progresse dans la bonne direction, mais il faudrait énormément d'étapes pour atteindre le minimum.
* Avec un taux **bien choisi**, on atteint le minimum en quelques étapes.
* Avec un taux **trop grand**, les pas sont si grands que l'on « saute » par-dessus la vallée, et l'on finit même par s'éloigner du minimum : on dit que l'algorithme **diverge**.

```{admonition} Pour les curieuses et les curieux
:class: dropdown, note

En notant $\mathcal{L}(w)$ la perte et $\rho$ (la lettre grecque « rho ») le taux d'apprentissage, une étape de descente de gradient s'écrit :

$$
    w_{t+1} = w_t - \rho \, \mathcal{L}'(w_t)
$$

où $\mathcal{L}'(w_t)$ est la dérivée de la perte (sa pente) au point $w_t$.
Le signe « moins » traduit le fait que l'on se déplace dans la direction **opposée** à la pente : si la courbe monte vers la droite (pente positive), on se déplace vers la gauche, et inversement.

Lorsqu'il y a plusieurs paramètres, la dérivée est remplacée par le **gradient**, qui regroupe la pente de la perte par rapport à chacun des paramètres.
```

## Récapitulatif

```{admonition} À retenir
:class: important

* Un **neurone** calcule une somme pondérée de ses entrées (avec des **poids** et un **biais**), puis applique une **fonction d'activation**. Un modèle constitué d'un seul neurone s'appelle un **perceptron**.
* Entraîner un modèle, c'est chercher les paramètres qui minimisent la **fonction de perte**.
* La **descente de gradient** fait cela pas à pas, en suivant la pente de la perte vers le bas.
* Le **taux d'apprentissage** règle la taille des pas : trop petit, l'apprentissage est lent ; trop grand, il diverge.
```

Le perceptron reste un modèle très limité.
Dans le [chapitre suivant](sec:mlp), nous verrons comment combiner de nombreux neurones pour obtenir des modèles beaucoup plus riches.

## Exercices

````{admonition} Exercice #1
Un neurone a deux entrées, de poids $w_1 = 3$ et $w_2 = -2$, et un biais $b = 0{,}5$. Sa fonction d'activation est l'identité.
Que vaut sa sortie lorsque les entrées valent $x_1 = 1$ et $x_2 = 2$ ?

```{admonition} Solution
:class: dropdown, tip

$3 \times 1 + (-2) \times 2 + 0{,}5 = 3 - 4 + 0{,}5 = -0{,}5$.
```
````

````{admonition} Exercice #2
Pendant l'entraînement d'un modèle, vous observez que la perte augmente d'une étape à l'autre, et prend des valeurs de plus en plus grandes.
Quel réglage modifieriez-vous en priorité, et dans quel sens ?

```{admonition} Solution
:class: dropdown, tip

C'est le symptôme typique d'un taux d'apprentissage **trop grand** : l'algorithme diverge. Il faut le diminuer.
```
````

````{admonition} Exercice #3
À un moment de la descente de gradient, la pente de la courbe de perte est **positive** à l'endroit où l'on se trouve. Faut-il augmenter ou diminuer la valeur du paramètre ?

```{admonition} Solution
:class: dropdown, tip

Une pente positive signifie que la perte augmente quand le paramètre augmente. Pour faire diminuer la perte, il faut donc **diminuer** le paramètre.
```
````
