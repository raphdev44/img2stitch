# img2stitch
Une application pour créer des motifs de broderie machine.

Image (PNG/JPG) -> fichier de broderie machine (.dst, .pes, .jef, .exp, .vp3...)

    pip install -e ".[web]"
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

## Algorithmes de numérisation
`--engine standard` (défaut) découpe l'image en masques de pixels et remplit ces masques rangée par rangée.
`--engine vector` (option « Par vectorisation (vpype) » du formulaire) passe par une géométrie :

1. contours extraits en suréchantillonnage puis simplifiés — les bords suivent les courbes au lieu de l'escalier de pixels ;
2. **compensation de tension** (*pull compensation*, `PULL_MM`) : chaque forme est dilatée pour déborder sous le fil voisin ;
3. **sous-couche** de stabilisation (contour intérieur piqué en longs points) sur les zones assez grandes ;
4. remplissage vectoriel : les rangées sont coupées exactement par le bord des formes ;
5. parcours ordonné au plus proche voisin puis inversé quand c'est plus court — l'équivalent de `linemerge` + `linesort`.

`--engine industrial` (option « Industriel » du formulaire) utilise aussi les formes vectorisées, avec un
parcours optimisé pour la production : regroupement des couleurs, remplissage en serpentin, déplacements
couverts et réduction des coupes de fil.

Le résultat vectorisé est plus régulier sur les formes courbes, au prix d'un peu plus de points. Le moteur
standard reste le défaut. `pip install -e ".[vector]"` installe la dépendance (`shapely`) requise par les deux
moteurs géométriques en mode CLI ; `pip install -e ".[web]"` installe aussi Shapely, Flask et toutes les
dépendances de base pour l'application web.
Les étapes utiles du document de spécification (vtracer + vpype) sont reproduites avec OpenCV et Shapely :
vpype épingle `Shapely==1.8.0` (plus de wheel depuis Python 3.11) et les wheels vtracer pour Python 3.14
plantent à l'appel — voir la docstring de `img2stitch/algorithm_vectorize.py`.

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
    pip install -e ".[web]"
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

### Broderie à la main
Depuis la galerie, « + Nouvelle broderie à la main » ouvre un import dédié qui reprend les
mêmes contrôles de couleurs et d'arrière-plan que le point de croix. Le moteur, lui, ne
remplit pas : il **émule des points de main**.

Chaque forme connexe d'une couleur est mesurée (axe médian, largeur au 80e centile,
longueur, rondeur, présence de trous) puis reçoit le point dont la géométrie raconte cette
forme :

- trait très fin (moins de 0,7 mm) : point arrière, en une seule ligne continue — jamais
  de tirets ; trait fin : arrière si la forme enferme une surface (liseré d'un dessin),
  point de tige sinon (ligne en torsade) ;
- bande de 2 à 4 mm : point de chaînette si elle est longue, point fendu sinon ;
- bande de 4 à 7,5 mm : point d'arête pour une feuille allongée, point passé sinon ;
- bande plus large : point passé jusqu'à 9 mm, puis semis ou reprisage au-delà — le
  remplissage reste à la marge, comme au fil et à l'aiguille ;
- petit galet rond : nœud français (moins de 3 mm²) ou roue enroulée ;
- grande surface ronde : semis de points, jamais un satin en éventail.

La bibliothèque compte vingt-trois points (`img2stitch/handstitch.py`) : point avant,
point arrière, point de tige, point fendu, chaînette, point posé, point de croix,
reprisage, point passé (long-and-short), point d'arête, point de filet, chevron,
fougère, plume, crétois, feston, graine, nœud français, point de poste, marguerite,
roue enroulée, toile d'araignée et tulipe. Tous sont dessinés par leur géométrie réelle
(boucles arrondies de la chaînette, bras de la fougère, spirale de la roue), avec un
léger hasard reproductible sur la longueur, l'angle et la position : c'est ce désordre
réglé qui distingue le travail main du point machine.

Le formulaire propose aussi de forcer une famille — tige, chaînette, feuilles, fleurs ou
croix — pour uniformiser les tracés. En automatique, le classificateur décide forme par
forme.

Une broderie à la main **n'est pas modifiable** : ni éditeur de zones, ni optimisation,
ni fusion de fils. Les points ont été choisis pour la forme de départ ; il faut recréer
un motif depuis l'image. Le motif est centré, limité à 110 mm sur son plus grand côté
(au-delà, les points perdent l'échelle de la main) et exporté dans les huit formats.
