# img2stitch
Une application pour créer des motifs de broderie machine.

Image (PNG/JPG) -> fichier de broderie machine (.dst, .pes, .jef, .exp, .vp3...)

    pip install -e .
    img2stitch logo.png logo.pes --width 80 --colors 4 --preview apercu

`--preview apercu` écrit DEUX images :
- `apercu_brut.png`  : tracé technique des points, avec les couleurs de fil (pour vérifier le parcours) ;
- `apercu_photo.png` : simulation réaliste de la broderie terminée (fils en relief, brillance, ombres, tissu tissé).
  Tissu au choix : `--fabric blanc|creme|gris|noir|marine|denim|rouge|#rrggbb`.

`--width` est la largeur de l'IMAGE entière (marges blanches comprises), pas celle du motif brodé.
Pour imposer la taille du motif brodé lui-même : `--max-width 100 --max-height 80` (en mm) ; le motif est alors
le plus grand possible tout en tenant strictement dans ce rectangle, proportions conservées.

`--background-scope outside` ne retire que le fond relié aux bords de l'image : les zones de la même couleur
à l'intérieur du motif sont brodées. Par défaut (`all`), toutes les zones de la couleur du fond sont retirées.

## Sauts et coupes de fil
Le parcours est optimisé pour limiter les sauts : un déplacement est piqué "à couvert" (sous une couleur
posée plus tard) quand c'est possible, sinon c'est un saut, précédé d'une coupe (TRIM) s'il dépasse
`--trim-threshold` (3 mm), avec points d'arrêt (désactivables : `--no-lock`).
`--max-visible-travel` règle le compromis : plus grand = moins de sauts mais plus de fil visible sur le remplissage.

```python
from img2stitch import convert, render_previews, Settings
pattern = convert("logo.png", "logo.pes", Settings(width_mm=70, n_colors=3))
render_previews(pattern, "apercu", fabric="marine")
```

Le motif est centré sur (0,0) (centre du cerceau). Tests : `pip install pytest && pytest`.
Exemple complet dans `examples/`.

## Application web
    pip install -e .[web]
    flask --app webapp run

Puis ouvrir http://127.0.0.1:5000 : création de motifs (image, nombre de couleurs, suppression du fond),
aperçus réaliste / simple, téléchargement dans tous les formats, galerie.
Les motifs sont stockés dans `instance/designs/` (modifiable via la variable `IMG2STITCH_DATA`).

### Pixel art en point de croix
Depuis la galerie, « + Nouveau pixel art en point de croix » ouvre un import dédié.
Un moteur distinct détecte les cases d'une grille régulière et génère une croix de 2 mm
par case colorée, sans broder les lignes du quadrillage. Les petites images à leur résolution
native sont également acceptées : un pixel correspond alors à une case.
Les dimensions restent fixes, à raison de 2 mm par case.
La création utilise les mêmes contrôles de couleurs que les motifs classiques :
curseur de 1 à 12, valeur proposée automatiquement, aperçu des palettes et pastilles
pour supprimer une couleur ou la rétablir. Une couleur supprimée laisse ses cases vides.
Le nombre choisi est un maximum : le moteur n'ajoute pas de couleurs si l'image en contient moins.
Les pastilles des fils permettent de changer leurs couleurs ; les fils devenus identiques
sont regroupés automatiquement. Les modifications régénèrent les aperçus et les exports,
sans modifier les dimensions des croix.
Les nuances très proches dues à la compression sont regroupées automatiquement.
La suppression de l'arrière-plan conserve les mêmes trois modes que la création classique.
Les aperçus et les huit formats de téléchargement restent disponibles.
Limites : 200 cases par côté (400 mm), 40 000 cases, 256 couleurs ; une grille non détectée
sur une grande image est signalée comme une erreur.
