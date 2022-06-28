# Fin de l'étude - Vérification des résultats

À ce stade, les résultats du traitement d'image effectué par le programme ont été fournis à la cliente.

Bien que celle-ci ait confirmé que la vérification visuelle des résultats serait
déjà satisfaisante, la convention d'étude a prévu une phase de test sur les
données disponibles en Open Data, et donc supposées aisément exploitables.

Afin de procéder à cette vérification, les liens suivants ont été fournis par la cliente : 



- Dataset 1 : [OPEN_DATA__deploiements-THD_2017_T4.xlsx](https://www.data.gouv.fr/fr/datasets/r/ff53e607-b463-452c-a136-44d919c55e1a)

- Dataset 2 : [Le marché du haut et très haut débit fixe (déploiements)](https://www.data.gouv.fr/fr/datasets/le-marche-du-haut-et-tres-haut-debit-fixe-deploiements/)


## Cadre de l'extraction de données

Pour rappel, les [cartes fournies dans le cadre de
l'Étude](https://github.com/yberreby/carto-res-arcep/blob/e9540baa0a2994560446c4c09fca7a2106be65f8/DEGROUPAGE_v2.pdf)
portent sur le statut de **dégroupage** des NRA (Noeuds de Raccordement ADSL) français.

À partir de ces cartes, il a été demandé de fournir des données donnant un
**statut de dégroupage par commune** de France métropolitaine.
Pour vérifier les résultats obtenus, il faut donc trouver de telles données en libre accès.

Note : la notion de dégroupage ne s'applique pas aux raccordements en **fibre optique**.


## Dataset 1

Le dataset est un fichier Excel composant de nombreux onglets, ainsi détaillés par les auteurs :

![](./d1-toc.png)

On trouve des données synthétiques (par département, période...) dans l'essentiel des onglets.
Seuls deux onglets traitent de communes individuelles : "Communes" et "FttH par communes".

Les données de l'onglet "Communes" ne donnent aucune information sur le dégroupage :

![](./d1-communes.png)

De même pour les données de "FttH par communes", puisqu'il s'agit là d'informations spécifiques à la fibre optique (FttH = _Fiber to the Home_).


## Dataset 2

Le dataset 2 est un ensemble de sous-datasets, dont un sous-ensemble est disponible pour chaque trimestre à partir de 2018 :
- `obs-hd-thd-deploiement` : il s'agit en fait de données quasi identiques à celles du dataset 1.
- `Departement`
- `Immeuble`
- `ZAPM` ("Zone Arrière du Point de Mutualisation", concept propre à la fibre optique)
- `Previsionnel`
- `Commune`

![](./d2-listing.png)


Seul les datasets `Commune` sont susceptibles d'être appropriés pour nos vérifications, puisqu'il nous faut un statut de dégroupage par commune.

Après décompression, celui-ci s'est avéré être composé de deux bases de données, l'une au format `DBF` et l'autre au format `shapefile`, accompagnées de métadonnées.

La lecture de la première révèle la présence des colonnes suivantes (**non documentées**) :

- `ID`: identifiant unique de l'entrée
- `INSEE_COM`: code INSEE de la commune
- `NOM_COM`: nom de la commune
- `INSEE_AR`: code INSEE de l'arrondissement
- `INSEE_DEP`: code INSEE du département
- `NOM_DEP`: nom du département
- `INSEE_REG`:  code INSEE de la région
- `NOM_REG`: nom de la région
- `NOM_REG_1`: un numéro de SIREN
- `POPULATION`: colonne manifestement toujours à 0
- `orig_ogcf`: inconnu
- `ZMD_privé`: ZMD = Zone Moins Dense
- `L_33-13`: a priori, opérateur engagé vis-à-vis de [l'article L. 33-13](https://fibre.guide/deploiement/zone-amii/article-l-33-13).
- `ZTD_2013`: inconnu
- `zone`: type de zone ("zipu", "zipri"...)
- `STATUT`: "Commune simple", "Préfecture"...
- `Dpt`: département
- `Locaux`: inconnu
- `couv`: inconnu
- `oi`: a priori "Opérateur d'Immeuble"
- `oi_bis`: similaire à `oi`, rôle inconnu
- `OI_AMEL`: inconnu
- `ftth`: inconnu - un chiffre relatif au déploiement FttH


J'ai visualisé les quelques colonnes inconnues ayant une chance de donner des informations utiles sur le dégroupage, mais il a rapidement été clair que celles-ci ne correspondent pas à celles utilisées pour produire les cartes fournies en amont de l'étude. De plus, l'absence de documentation compromet une éventuelle transformation de ces données pour essayer de reconstruire celles désirées : on ne sait pas ce qu'on manipule avec certitude.
 
![](./oi.png)

![](./couv.png)



## Recherches supplémentaires

En plus des données fournies par la cliente, j'ai cherché des données relatives au dégroupage sur [le site de l'ARCEP](https://www.arcep.fr/cartes-et-donnees.html) et [data.gouv.fr](https://www.data.gouv.fr/fr/), mais n'ai rien trouvé qui puisse être exploitable d'office, sans nécessiter des transformations qui prendraient un temps non disponible dans le cadre de l'étude, et devant _elles-mêmes_ être validées pour être méthodologiquement valide.

