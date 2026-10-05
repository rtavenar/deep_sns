"""Convertit les chapitres MyST (content/*.md) en notebooks pour Google Colab.

Usage : python scripts/make_notebooks.py <dossier_de_sortie>

- les directives MyST (encadrés, images...) sont traduites en Markdown simple,
  lisible dans Colab ;
- une première cellule récupère les données et les utilitaires du livre
  lorsque le notebook est exécuté dans Colab.
"""
import re
import shutil
import sys
from pathlib import Path

import jupytext
import nbformat

REPO = "https://github.com/rtavenar/deep_sns.git"
BOOK_URL = "https://rtavenar.github.io/deep_sns"

SETUP_CODE = f"""# Cellule de préparation : à exécuter en premier
# (dans Google Colab, elle récupère les données utilisées dans ce chapitre)
import sys
if "google.colab" in sys.modules:
    !git clone -q --depth 1 -b notebooks {REPO} /content/deep_sns
    %cd /content/deep_sns/content"""

FENCE = re.compile(r"^(`{3,}|:{3,})\s*(\{([\w-]+)\})?\s*(.*)$")


def myst_to_markdown(text, chapter_titles):
    out = []
    stack = []  # pile de (fence, kind)
    for line in text.splitlines():
        m = FENCE.match(line)
        if m:
            fence, _, directive, arg = m.groups()
            # fermeture d'un bloc ouvert
            if stack and directive is None and not arg and fence == stack[-1][0]:
                _, kind = stack.pop()
                if kind == "code":
                    out.append(line)
                else:
                    out.append("")
                continue
            if directive is None:  # bloc de code ordinaire (```python)
                stack.append((fence, "code"))
                out.append(line)
                continue
            if directive in ("image", "figure"):
                stack.append((fence, "image"))
                out.append(f"![]({arg})")
                continue
            # encadré (admonition, tip, note...)
            title = arg or directive.capitalize()
            stack.append((fence, "admonition"))
            out.append("")
            out.append(f"> **{title}**")
            out.append(">")
            continue
        # options de directive (:class: ..., :width: ...)
        if stack and stack[-1][1] in ("admonition", "image") and re.match(r"^:[\w-]+:", line):
            continue
        if stack and stack[-1][1] == "image":
            continue
        if stack and stack[-1][1] == "admonition":
            out.append(f"> {line}" if line.strip() else ">")
            continue
        out.append(line)
    text = "\n".join(out)
    # étiquettes (sec:xxx)= et références croisées
    text = re.sub(r"^\(sec:[\w-]+\)=\s*$", "", text, flags=re.M)
    text = re.sub(r"\[\]\((sec:[\w-]+)\)",
                  lambda m: f"« {chapter_titles.get(m.group(1), 'chapitre')} »", text)
    text = re.sub(r"\[([^\]]+)\]\(sec:[\w-]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\((\w+)\.md\)", rf"[\1]({BOOK_URL}/content/\2.html)", text)
    return text


def main(out_dir):
    src = Path("content")
    out = Path(out_dir) / "content"
    out.mkdir(parents=True, exist_ok=True)

    # titres des chapitres, pour les références croisées
    titles = {}
    for md in src.glob("*.md"):
        t = md.read_text(encoding="utf-8")
        lbl = re.search(r"^\((sec:[\w-]+)\)=\s*\n+#\s+(.+)$", t, flags=re.M)
        if lbl:
            titles[lbl.group(1)] = lbl.group(2).strip()

    for md in sorted(src.glob("*.md")):
        if "kernelspec" not in md.read_text(encoding="utf-8"):
            continue  # pages sans code (intro, glossaire)
        nb = jupytext.read(md)
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                cell.source = myst_to_markdown(cell.source, titles)
            cell.metadata.pop("tags", None)
        nb.cells.insert(0, nbformat.v4.new_code_cell(SETUP_CODE))
        nb.metadata["kernelspec"] = {"display_name": "Python 3",
                                     "language": "python", "name": "python3"}
        nb.metadata.pop("jupytext", None)
        nbformat.write(nb, out / f"{md.stem}.ipynb")
        print("->", out / f"{md.stem}.ipynb")

    for extra in ["data", "img"]:
        if (src / extra).exists():
            shutil.copytree(src / extra, out / extra, dirs_exist_ok=True)
    shutil.copy(src / "notebook_utils.py", out / "notebook_utils.py")
    (Path(out_dir) / "README.md").write_text(
        "Notebooks générés automatiquement à partir des sources du livre "
        f"(branche `main`) pour être ouverts dans Google Colab.\n\nLivre : {BOOK_URL}\n",
        encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "_build/notebooks")
