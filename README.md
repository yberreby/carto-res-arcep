# Télécom Étude - Étude 221038 - "Extraction d’informations sur des cartes de basse qualité"

## Fichiers externes

Les fichiers suivants ont été obtenus de sources externes :

- `DEGROUPAGE_v2.pdf` : fourni par la cliente
- [`communes-departements-regions.csv`](https://www.data.gouv.fr/en/datasets/communes-de-france-base-des-codes-postaux/)
- [`villes_france.csv`](https://sql.sh/ressources/sql-villes-france/villes_france.sql)


## Méthodologie suivie

### Conversion PDF -> PNG

Effectuée avec `./export.sh`. Crée et remplit le répertoire `raw`, qui contient
alors une image PNG par page du document `DEGROUPAGE_v2.pdf`.

### Détourage de la France métropolitaine et rognage des images.

Réalisé à la main avec GIMP.
Les images résultantes doivent être stockées dans le répertoire `processed`.

Pour chaque image `raw/map-XX.png`, dans GIMP :

- Entourer la France métropolitaine avec l'outil "lasso"
- `Image > Crop to Selection`
- Clic droit sur couche active, `Add alpha channel`
- `Select > Invert`
- Appui sur la touche `Del`
- Sauvegarde de `processed/map-XX.xcf`
- Export en PNG vers `processed/map-XX.png` (`Save gamma` coché, autres réglages par défaut)

Les images finales sont donc `processed/map-01.png`, `processed/map-02.png`, ...,
`processed/map-14.png`.

Elles ont été mises à la disposition de la cliente hors GitHub. 

### Mise en cohérence des systèmes de coordonnées

Afin d'aligner les images pré-traitées avec GIMP de manière reproductible et
programmable, il a fallu choisir des **points de contrôle géographiques**. De tels
points doivent être suffisamment "pointus" pour être précisément repérables,
mais suffisamment grands pour apparaître sur un maximum de cartes.

### Relevés manuels

Sept points de contrôle ont été choisis. Leurs coordonnées GPS (colonnes `lat`
et `lng`) ont été relevées et stockées dans `gps.csv`.  On y a associé un nom
(colonne `nom`) afin de les identifier de manière unique.

Les coordonnées `(x,y)` de chacun de ces points dans le repère
intrinsèque à chaque image ont ensuite été relevées pour chacune des images et
stockées dans `ref.csv`, suivant la structure suivante :

|    `nom`           | `image`                    | `x_image`       | `y_image`       |
| -------------------| ---------------------------| ----------------| ----------------|
| point de contrôle  | numéro de l'image (1 à 14) | coordonnée en x | coordonnée en y |

NB: L'axe x est horizontal et croissant de gauche à droite. L'axe y est vertical
**et croissant de haut en bas**. Ces coordonnées sont exprimées **en pixels**.


### Calculs

La projection Lambert93 étant la [projection cartographique officielle pour les cartes de France métropolitaine](https://www.legifrance.gouv.fr/affichTexte.do?cidTexte=LEGITEXT000005630333),
on fait l'hypothèse que celle-ci a été utilisée pour réaliser les cartes de
l'ARCEP.

On utilise la bibliothèque `pyproj` pour transformer les coordonnées de
`gps.csv` en coordonnées dans le système Lambert93. Grâce aux relevés de
`ref.csv`, on résoud ensuite un système d'équations linéaires afin d'exprimer la
transformation affine permettant de passer du système Lambert93 au repère
intrinsèque à chaque image.

Ceci nous permet d'**associer à chaque ville la section d'image correspondante**.

On en choisit les dimensions de manière à couvrir la plus petite zone permettant
de garder une marge d'erreur (les sources d'erreur comportant : le
positionnement des points dans les cartes ARCEP, les coordonnées GPS des villes
fournies dans le fichier utilisé comme référence, les pointages manuels en
coordonnées intrinsèques & GPS des points de contrôle utilisés pour
l'alignement) : un côté de 30 pixels a été retenu.


### Classification des sections d'image

Une fois une section extraite, il s'agit de la classer.

Les marqueurs des cartes étant différentiables par leurs couleurs, on choisit de
se fonder sur celles-ci. Pour cela, on cherche une **couleur représentative** de
chaque section de carte.

Afin d'obtenir une telle couleur, un moyennage en RGB des couleurs des pixels
d'une section a aisément donné des résultats corrects dans les cas simples, mais
eut plus de difficultés avec les sections à faible contraste, à proximité de
frontières de départements, ou lorsqu'un point coloré (= avec couverture haut
débit) est entouré de nombreux points clairs. On retient une approche hybride,
utilisant l'espace de couleurs [HSV](https://en.wikipedia.org/wiki/HSL_and_HSV)
afin de rejeter les pixels trop sombres et de favoriser les couleurs les plus
vives. Pour détails, voir la fonction `patch_repr_color` dans `etude.py`.

Une fois les couleurs représentatives obtenues, on les compare à des couleurs de référence. Celles-ci sont spécifiques à chaque carte et stockées dans `colors.csv` :


|    `im`            | `degroupe`                 | `r`, `g`, `b`|
| -------------------| ---------------------------| -------------|
| numéro de l'image  | classe correspondante      | couleur RGB  |

Les classes sont : 
- `1`  pour une couverture haut débit
- `0` pour une absence de couverture
- `-1` pour une couverture ambiguë.


Grâce à ces relevés, on utilise la [méthode du plus proche voisin](https://fr.wikipedia.org/wiki/M%C3%A9thode_des_k_plus_proches_voisins) pour classer chaque couleur représentative.

Les résultats sont stockés dans [`resultats.csv`](https://drive.google.com/file/d/1EeXeSPHfj30dl8UsG6-nvp4MSIwOlxck/view?usp=sharing), où, pour chaque ligne (=
ville), la colonne `couverture-im-XX` donne la classification de cette ville
dans l'image numéro `XX`.

Les images 13 et 14 (les plus récentes) ont été omises des résultats finaux, car
leur faible contraste rend fragile la méthode de classification utilisée pour
les autres images, et les données correspondantes sont disponibles en Open Data
auprès de l'ARCEP.

Une vérification visuelle des résultats est possible : [lien Google Drive](https://drive.google.com/drive/folders/1gl27XERIBwZzOGFeMO2O5ZR7VbPPm1yP?usp=sharing).


## Résumé des rendus finaux

- [Code source du projet](https://github.com/yberreby/carto-res-arcep)
- [Données produites par le programme](https://drive.google.com/drive/folders/1QSDqwGM51KmNDY8i_JgQVheoVwiJ4pGz?usp=sharing)


## Usage du code

- Les dépendances Python du projet sont spécifiées dans `requirements.txt`.
- Exécuter le notebook `produce-report.ipynb` permet de regénérer `resultats.csv`.
- Le notebook `check.ipynb` permet de contrôler visuellement l'alignement des
différentes images.

L'usage de Jupyter **Lab** est recommandé, afin de pouvoir zoomer de manière
interactive sur les images dans les notebooks.
