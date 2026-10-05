# Glossaire

Les termes sont classés par ordre alphabétique. Le terme anglais, que l'on rencontre dans la plupart des ressources en ligne et dans la documentation de `keras`, est indiqué entre parenthèses.

**Activation (fonction d')** (_activation function_)
: Transformation appliquée par un neurone au résultat de sa somme pondérée. Exemples : ReLU, sigmoïde, tanh, softmax. Voir [](sec:mlp).

**Apprentissage par transfert** (_transfer learning_)
: Réutilisation d'un modèle pré-entraîné comme point de départ pour un nouveau problème, en remplaçant sa tête de classification. Voir [](sec:cnn).

**Architecture**
: La « forme » d'un réseau de neurones : nombre de couches, nombre de neurones par couche, fonctions d'activation.

**Biais** (_bias_)
: Paramètre d'un neurone qui s'ajoute à la somme pondérée de ses entrées.

**Canal** (_channel_)
: Une des « couches » de nombres qui composent une image : 1 canal pour une image en noir et blanc, 3 (rouge, vert, bleu) pour une image en couleur.

**Carte d'activation** (_feature map_, _activation map_)
: Image produite par l'application d'un filtre de convolution ; elle indique où se trouve le motif détecté par le filtre.

**Classification** (_classification_)
: Problème dans lequel la cible est une catégorie parmi un nombre fini de possibilités (les **classes**). On parle de classification **binaire** quand il y a deux classes.

**Convolution** (_convolution_)
: Opération qui fait glisser un petit filtre sur une image et calcule, à chaque position, une somme pondérée des pixels recouverts. Voir [](sec:cnn).

**Couche cachée** (_hidden layer_)
: Couche de neurones située entre la couche d'entrée et la couche de sortie.

**Couche dense** (_dense layer_, _fully-connected layer_)
: Couche dont chaque neurone est relié à tous les neurones de la couche précédente.

**Descente de gradient** (_gradient descent_)
: Méthode d'optimisation qui modifie les paramètres pas à pas, en suivant la pente de la fonction de perte vers le bas. Voir [](sec:perceptron).

**Entraînement** (_training_)
: Recherche des valeurs des paramètres qui minimisent la fonction de perte sur le jeu d'entraînement. Synonyme : apprentissage, optimisation.

**Filtre** (_filter_, _kernel_)
: Petit tableau de nombres (par exemple 3 × 3) utilisé dans une convolution. Ses valeurs sont des paramètres appris.

**Fine-tuning**
: Ré-entraînement léger de toutes les couches d'un modèle pré-entraîné sur un nouveau jeu de données.

**Flatten** (« aplatir »)
: Couche qui met toutes les valeurs des cartes d'activation bout à bout, pour pouvoir les donner à des couches denses.

**Gradient** (_gradient_)
: Ensemble des pentes de la fonction de perte par rapport à chacun des paramètres. Il indique dans quelle direction la perte augmente le plus vite.

**Hyperparamètre** (_hyperparameter_)
: Réglage choisi à l'avance par la personne qui construit le modèle, et non appris à partir des données. Exemples : le taux d'apprentissage, le nombre de couches cachées, le nombre de neurones par couche.

**Jeu d'entraînement / jeu de test** (_training set / test set_)
: Partie des données utilisée pour régler les paramètres / partie mise de côté pour évaluer le modèle sur des exemples qu'il n'a jamais vus.

**MLP, perceptron multicouche** (_Multi-Layer Perceptron_)
: Réseau de neurones formé d'une couche d'entrée, d'une ou plusieurs couches cachées denses et d'une couche de sortie.

**Neurone** (_neuron_, _unit_)
: Unité de calcul de base d'un réseau : calcule une somme pondérée de ses entrées, ajoute un biais, puis applique une fonction d'activation.

**Padding** (« rembourrage »)
: Ajout d'une bordure de zéros autour d'une image avant une convolution, pour que la carte d'activation garde la même taille que l'image.

**Paramètres** (_parameters_, _weights_)
: Les valeurs réglées pendant l'apprentissage : poids et biais de tous les neurones.

**Perceptron** (_perceptron_)
: Modèle constitué d'un seul neurone.

**Perte (fonction de)** (_loss function_)
: Mesure de l'erreur commise par le modèle sur un ensemble d'exemples. On cherche à la rendre la plus petite possible. Exemple : l'erreur quadratique moyenne (_Mean Squared Error_, MSE) en régression.

**Pixel** (_pixel_)
: Un des petits carrés qui composent une image, décrit par un nombre par canal.

**Poids** (_weight_)
: Paramètre qui multiplie une entrée d'un neurone.

**Pooling** (« regroupement »)
: Couche qui réduit la taille des cartes d'activation en résumant chaque petit carré de pixels, par exemple par sa valeur maximale (_max pooling_).

**Pré-entraîné (modèle)** (_pre-trained model_)
: Modèle déjà entraîné sur une grande base de données (par exemple ImageNet) et mis à disposition pour être réutilisé.

**Réseau convolutif** (_Convolutional Neural Network_, CNN, ConvNet)
: Réseau de neurones composé de couches de convolution et de pooling, suivies d'une tête de classification. Voir [](sec:cnn).

**Régression** (_regression_)
: Problème dans lequel la cible est un nombre.

**ReLU** (_Rectified Linear Unit_)
: Fonction d'activation qui remplace les valeurs négatives par 0 et laisse les valeurs positives inchangées. Choix par défaut pour les couches cachées.

**Sigmoïde** (_sigmoid_)
: Fonction d'activation qui produit une valeur entre 0 et 1. Utilisée en sortie pour la classification binaire.

**Softmax** (_softmax_)
: Fonction d'activation qui transforme une liste de scores en probabilités (positives, de somme 1). Utilisée en sortie pour la classification à plusieurs classes.

**Taux d'apprentissage** (_learning rate_)
: Hyperparamètre qui règle la taille des pas de la descente de gradient.

**Tête de classification** (_classification head_)
: Dernières couches (denses) d'un réseau convolutif, qui produisent la prédiction à partir des cartes d'activation.

**Variable cible** (_target_)
: Ce que l'on cherche à prédire, notée $y$. La prédiction du modèle est notée $\hat{y}$.

**Variables explicatives** (_features_)
: Les informations dont dispose le modèle pour faire sa prédiction, notées $x$. On parle aussi d'entrées ou de caractéristiques.
