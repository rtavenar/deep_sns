"""Petites fonctions utilitaires pour les figures du livre."""
import matplotlib.pyplot as plt
from matplotlib import rc

BLEU = "#4C72B0"
CYAN = "#64B5CD"
VERT = "#55A868"
ORANGE = "#DD8452"
GRIS = "#8C8C8C"


def prepare_notebook_graphics():
    plt.ion()
    rc("font", size=14)


def draw_network(layer_sizes, layer_names=None, layer_colors=None,
                 weight_labels=None, ax=None, figsize=(8, 4.5)):
    """Dessine un réseau de neurones entièrement connecté.

    layer_sizes : nombre de neurones par couche (entrée -> sortie)
    layer_names : titre affiché au-dessus de chaque couche
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    n_layers = len(layer_sizes)
    if layer_colors is None:
        layer_colors = [BLEU] + [CYAN] * (n_layers - 2) + [VERT]
    max_size = max(layer_sizes)
    positions = []
    for i, n in enumerate(layer_sizes):
        x = i * 3.
        ys = [(max_size - 1) / 2. - j + (n - 1) / 2. - (max_size - 1) / 2.
              for j in range(n)]
        ys = [y + (max_size - 1) / 2. for y in ys]
        positions.append([(x, y) for y in ys])
    # connexions
    for i in range(n_layers - 1):
        for (x0, y0) in positions[i]:
            for (x1, y1) in positions[i + 1]:
                ax.plot([x0, x1], [y0, y1], color=GRIS, lw=.6, zorder=1)
    # neurones
    for i, layer in enumerate(positions):
        for (x, y) in layer:
            ax.add_patch(plt.Circle((x, y), .3, facecolor=layer_colors[i],
                                    edgecolor="black", zorder=2))
        if layer_names is not None:
            ax.text(layer[0][0], max_size - .1, layer_names[i],
                    ha="center", va="bottom", fontsize=12)
    if weight_labels is not None:
        for i, lbl in enumerate(weight_labels):
            ax.text(i * 3. + 1.5, (max_size - 1) / 2., lbl, ha="center",
                    va="center", fontsize=12, zorder=3,
                    bbox=dict(facecolor="white", edgecolor="none"))
    ax.set_xlim(-.8, (n_layers - 1) * 3. + .8)
    ax.set_ylim(-.8, max_size + .6)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def draw_neuron(ax=None, figsize=(8, 4)):
    """Dessine un neurone unique : entrées, poids, somme, activation, sortie."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    inputs = ["$x_1$", "$x_2$", "$x_3$", "$+1$"]
    weights = ["$w_1$", "$w_2$", "$w_3$", "$b$"]
    ys = [3, 2, 1, 0]
    for lbl, w, y in zip(inputs, weights, ys):
        ax.add_patch(plt.Circle((0, y), .35, facecolor=BLEU if lbl != "$+1$"
                                else "white", edgecolor="black", zorder=2))
        ax.text(0, y, lbl, ha="center", va="center", fontsize=13, zorder=3,
                color="white" if lbl != "$+1$" else "black")
        ax.annotate("", xy=(3.6, 1.5), xytext=(.35, y),
                    arrowprops=dict(arrowstyle="->", color=GRIS))
        ax.text(1.9, y + (1.5 - y) * .48 + .15, w, fontsize=13,
                ha="center", color="black")
    ax.add_patch(plt.Circle((4, 1.5), .45, facecolor="white",
                            edgecolor="black", zorder=2))
    ax.text(4, 1.5, r"$\Sigma$", ha="center", va="center", fontsize=16)
    ax.annotate("", xy=(6.1, 1.5), xytext=(4.45, 1.5),
                arrowprops=dict(arrowstyle="->", color=GRIS))
    ax.text(5.3, 1.75, "activation", ha="center", fontsize=11)
    ax.add_patch(plt.Circle((6.5, 1.5), .4, facecolor=VERT,
                            edgecolor="black", zorder=2))
    ax.text(6.5, 1.5, "$a$", ha="center", va="center", fontsize=14,
            color="white")
    ax.text(0, 3.8, "entrées", ha="center", fontsize=12)
    ax.text(4, 2.3, "somme\npondérée", ha="center", fontsize=11)
    ax.text(6.5, 2.2, "sortie", ha="center", fontsize=12)
    ax.set_xlim(-.6, 7.2)
    ax.set_ylim(-.6, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
