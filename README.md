# Deep learning : les bases

Version simplifiée, en français uniquement, des notes de cours
[Introduction au Deep Learning](https://rtavenar.github.io/deep_book/),
destinée à un public non spécialiste en mathématiques ni en programmation.
Elle couvre le perceptron, le perceptron multicouche (MLP) et les réseaux convolutifs.

## Construire le livre

```bash
pip install -r requirements.txt
make html          # résultat dans _build/html/index.html
```

Les cellules de code sont exécutées à la construction (backend `keras` :
`torch` par défaut, modifiable via `KERAS_BACKEND=jax make html`).

Contrairement au livre complet, aucune dépendance LaTeX/TikZ n'est nécessaire
pour la version HTML : les schémas de réseaux sont dessinés avec matplotlib
(voir `content/notebook_utils.py`).
