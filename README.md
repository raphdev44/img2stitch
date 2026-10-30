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

## Deux algorithmes de numérisation
`--engine standard` (défaut) découpe l'image en masques de pixels et remplit ces masques rangée par rangée.
`--engine vector` (option « Par vectorisation (vpype) » du formulaire) passe par une géométrie :

1. contours extraits en suréchantillonnage puis simplifiés — les bords suivent les courbes au lieu de l'escalier de pixels ;
2. **compensation de tension** (*pull compensation*, `PULL_MM`) : chaque forme est dilatée pour déborder sous le fil voisin ;
3. **sous-couche** de stabilisation (contour intérieur piqué en longs points) sur les zones assez grandes ;
4. remplissage vectoriel : les rangées sont coupées exactement par le bord des formes ;
5. parcours ordonné au plus proche voisin puis inversé quand c'est plus court — l'équivalent de `linemerge` + `linesort`.

Le résultat est plus régulier sur les formes courbes, au prix d'un peu plus de points. Le moteur standard
reste le défaut. `pip install -e .[vector]` installe la dépendance (`shapely`).
Les étapes utiles du document de spécification (vtracer + vpype) sont reproduites avec OpenCV et Shapely :
vpype épingle `Shapely==1.8.0` (plus de wheel depuis Python 3.11) et les wheels vtracer pour Python 3.14
plantent à l'appel — voir la docstring de `img2stitch/vectorize.py`.

## Sauts et coupes de fil
Le parcours est optimisé pour limiter les sauts : un déplacement est piqué "à couvert" (sous une couleur
posée plus tard) quand c'est possible, sinon c'est un saut, précédé d'une coupe (TRIM) s'il dépasse
`--trim-threshold` (3 mm), avec points d'arrêt (désactivables : `--no-lock`).
`--max-visible-travel` règle le compromis : plus grand = moins de sauts mais plus de fil visible sur le remplissage.

```python
from img2stitch import convert, render_previews, Settings
from img2stitch.optimize import optimize_pattern
from img2stitch.radical import optimize_pattern_radical

pattern = convert("logo.png", "logo.pes", Settings(width_mm=70, n_colors=3))
render_previews(pattern, "apercu", fabric="marine")

motif, rapport = optimize_pattern(pattern)                 # passes prudentes
motif, rapport = optimize_pattern_radical(pattern)         # parcours réorganisé
```

Le motif est centré sur (0,0) (centre du cerceau). Tests : `pip install pytest && pytest`.
Exemple complet dans `examples/`.

## Application web
    pip install -e .[web]
    flask --app webapp run

Puis ouvrir http://127.0.0.1:5000 : création de motifs (image, nombre de couleurs, suppression du fond),
aperçus réaliste / simple, téléchargement dans tous les formats, galerie.
Les motifs sont stockés dans `instance/designs/` (modifiable via la variable `IMG2STITCH_DATA`).
Sur un motif classique prêt, le bouton **Optimiser le motif** ouvre un menu avec deux optimisations.
Les deux **dupliquent le motif** (même image, mêmes réglages) et n’optimisent que la copie : le motif
d’origine reste intact dans la galerie. La copie est nommée « nom - optimisé » (simple) ou
« nom - optimisé+ » (radicale) et l’application bascule dessus une fois la copie créée ; ses exports
et aperçus sont propres à la version optimisée.

**Optimisation simple** (conservatrice) : nettoie les points identiques et les micro-points (avec un
garde-fou de 25 % par séquence), fusionne les points alignés, limite les points piqués à 12 mm et
supprime les coupes avant les déplacements de 2,5 mm ou moins. L’ordre des objets et la densité des
remplissages sont conservés.

**Optimisation radicale** : réorganise en profondeur le parcours de la copie pour un usage direct en
machine. Elle enchaîne :

1. un nettoyage agressif des passages (doublons, micro-points, points alignés fusionnés par un
   ré-échantillonnage le long du tracé, avec conservation des coins) ;
2. la suppression des points recouverts par une couche brodée plus tard (points invisibles),
   vérifiée passage par passage sur la couverture réellement visible ;
3. le **réordonnancement des blocs et des passages** : blocs d’un même fil rapprochés, ordre des
   couleurs choisi au plus court, passages d’une couleur enchaînés au plus proche voisin puis
   améliorés par 2-opt, avec inversion du sens de piquage quand c’est plus court ;
4. la **suppression des coupes** : les déplacements courts deviennent des sauts, et un déplacement
   entièrement caché sous une zone brodée plus tard (ou sous le même fil) est **piqué** plutôt que
   coupé, par morceaux de 2 mm ;
5. un contrôle final : le rendu couleur du motif optimisé est comparé à l’original. Si la
   réorganisation change trop l’image (superposition, géométrie), elle est annulée et seules les
   passes de points sont gardées.

Les superpositions sont préservées : deux blocs qui se recouvrent gardent leur ordre relatif. Les
changements de fil consécutifs (arrêt opérateur) et les commandes non gérées (appliqué, sequins)
font abandonner l’optimisation radicale plutôt que de les perdre. Les gains mesurés sur les motifs
de test (7 motifs réels et 183 motifs importés) sont en moyenne de **−15 % de points** et **−60 % de
coupes**, avec une image identique à moins de 0,2 %.

La limite de 12 mm est une valeur de base, pas une certification pour une machine particulière : vérifiez
les recommandations du fabricant et contrôlez l’aperçu avant de broder.
Les traitements longs affichent une barre de progression.
Dans les deux parcours, supprimer une couleur réduit les fils utilisés sans retirer de zone
du motif : ses pixels sont réaffectés à la couleur conservée la plus proche en CIELAB.
Seuls la transparence et les réglages de suppression du fond créent des zones non brodées.

### Pixel art en point de croix
Depuis la galerie, « + Nouveau pixel art en point de croix » ouvre un import dédié.
Un moteur distinct détecte les cases d'une grille régulière et génère une croix de 2 mm
par case colorée, sans broder les lignes du quadrillage. Les petites images à leur résolution
native sont également acceptées : un pixel correspond alors à une case.
Les pixel arts agrandis sans quadrillage sont aussi acceptés si leurs blocs carrés sont
bien marqués et alignés régulièrement. Le moteur détecte automatiquement la taille et
le décalage des blocs, puis prélève leur couleur au centre, sans ajouter de lignes.
Les marges et la transparence sont conservées ; les arrondis de redimensionnement et
les petites variations dues au JPEG sont tolérés. Une image sans trame identifiable
ne permet pas de déduire automatiquement la taille des cases.
Les dimensions restent fixes, à raison de 2 mm par case.
La création utilise les mêmes contrôles de couleurs que les motifs classiques :
curseur de 1 à 12, valeur proposée automatiquement, aperçu des palettes et pastilles
pour supprimer une couleur ou la rétablir. Les cases d'une couleur supprimée sont
brodées avec la couleur conservée la plus proche (distance perceptuelle CIELAB),
sans perdre de croix. Il faut conserver au moins une couleur. Les cases transparentes
et le fond supprimé restent vides.
Le nombre choisi est un maximum : le moteur n'ajoute pas de couleurs si l'image en contient moins.
Les pastilles des fils permettent de changer leurs couleurs ; les fils devenus identiques
sont regroupés automatiquement. Les modifications régénèrent les aperçus et les exports,
sans modifier les dimensions des croix.
Les nuances très proches dues à la compression sont regroupées automatiquement.
La suppression de l'arrière-plan conserve les mêmes trois modes que la création classique.
Les aperçus et les huit formats de téléchargement restent disponibles.
Limites : 200 cases par côté (400 mm), 40 000 cases, 256 couleurs ; l'absence de grille
ou de blocs réguliers détectables sur une grande image est signalée comme une erreur.
