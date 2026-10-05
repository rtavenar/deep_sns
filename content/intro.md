# Deep learning : les bases

**par Romain Tavenard**

Ces notes accompagnent le cours d'introduction au _deep learning_ (apprentissage profond) donné à l'Université Rennes 2.
Elles sont une version **simplifiée** de notes de cours plus complètes ([disponibles ici](https://rtavenar.github.io/deep_book/)) :
elles sont pensées pour des lectrices et lecteurs qui ne sont spécialistes ni des mathématiques, ni de la programmation.

## Ce que vous allez apprendre

À la fin de ces notes, vous saurez :

* ce que signifie « apprendre à partir de données » ;
* ce qu'est un **neurone artificiel** et comment on règle ses paramètres grâce à la **descente de gradient** ;
* comment on assemble des neurones en couches pour former un **perceptron multicouche** (_Multi-Layer Perceptron_, MLP), le réseau de neurones le plus simple ;
* comment fonctionnent les **réseaux convolutifs**, les modèles de référence pour traiter des images ;
* comment déclarer ces réseaux en Python avec la bibliothèque `keras`.

## Comment lire ces notes

Les chapitres sont courts et se lisent dans l'ordre :

1. [Apprendre à partir de données](donnees.md) : le vocabulaire de base ;
2. [Un premier neurone : le perceptron](perceptron.md) : un modèle minimal, et la façon dont on l'entraîne ;
3. [Empiler des neurones : le perceptron multicouche](mlp.md) : notre premier vrai réseau de neurones ;
4. [Traiter des images : les réseaux convolutifs](convnets.md) : des réseaux adaptés aux images.

Un [glossaire](glossaire.md) rassemble les termes importants, avec leur traduction anglaise (la plupart des ressources en ligne sont en anglais).

Vous rencontrerez trois types d'encadrés :

```{admonition} À retenir
:class: important
Les idées essentielles du chapitre.
```

```{admonition} Pour les curieuses et les curieux
:class: dropdown, note
Des compléments plus mathématiques, **facultatifs** : on peut tout à fait suivre le cours sans les lire.
Cliquez sur le titre de l'encadré pour le déplier.
```

```{admonition} Exercice
Des petites questions pour vérifier que vous avez compris. La solution est donnée juste en dessous, dans un encadré à déplier : essayez de répondre avant de la regarder !
```

## Et le code ?

Ces notes contiennent un peu de code Python.
**Il n'est pas nécessaire de comprendre chaque ligne** : le code sert surtout à produire les figures et à montrer à quoi ressemble, concrètement, la déclaration d'un réseau de neurones.
Les cellules dont le code est masqué peuvent être ignorées sans problème.

Pour les travaux pratiques, nous utiliserons [Google Colab](https://colab.research.google.com), qui permet d'exécuter du Python dans un navigateur web, sans rien installer sur son ordinateur.

## Ce qu'il faut savoir avant de commencer

Très peu de choses :

* savoir lire un tableau de données (des lignes, des colonnes) ;
* savoir ce qu'est la **pente** d'une courbe (est-ce qu'elle monte ou est-ce qu'elle descend, et à quel point ?) ;
* avoir déjà vu un peu de Python (une variable, un appel de fonction).
