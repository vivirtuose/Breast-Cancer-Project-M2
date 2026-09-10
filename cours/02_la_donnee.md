# Module 2 — La donnée : du film scanné au tenseur

*Niveau : praticien. Prérequis : module 1.*

---

## 1. Ancrage

Un pixel de mammographie n'est pas une couleur : c'est le résultat d'une **mesure physique**, passée par une chaîne de conversions dont chaque étape peut détruire le signe que tu cherches.

---

## 2. Intuition

Une photo encode une **apparence** : la valeur d'un pixel n'a de sens que relativement aux autres pixels de la même photo. Une image médicale encode une **mesure** : la valeur d'un pixel prétend avoir un sens absolu, comparable d'une image à l'autre, d'un patient à l'autre.

L'analogie : un **thermomètre** contre une **photo de thermomètre**. Sur la photo, tu peux augmenter la luminosité sans rien casser — l'information est dans les formes. Sur la mesure, changer l'échelle change la valeur mesurée : 37 °C devient 40 °C et le patient est mort.

CBIS-DDSM est un cas hybride et instable, et c'est ce qui le rend piégeux : à l'origine c'est un **film argentique** — donc une mesure physique, avec une vraie unité — mais ce film a été **scanné**, par quatre appareils différents, dans quatre hôpitaux, chacun avec sa propre courbe de réponse. La chaîne de curation de 2017 a passé sa vie à recoller ces mesures sur une échelle commune. Le miroir JPEG que tu comptes utiliser défait une partie de ce travail.

---

## 3. Le mécanisme

### 3.1 Du film au nombre : la densité optique

Reprends la physique du module 1. Les rayons X traversent le sein, puis frappent le film. **Le film noircit là où il reçoit beaucoup de rayons X.**

Déroule la conséquence :

| Tissu | Atténuation | Rayons X arrivant au film | Film | Aspect à l'écran |
|---|---|---|---|---|
| Air (hors du sein) | nulle | maximum | très noirci | **noir** |
| Graisse | faible | beaucoup | noirci | sombre |
| Fibroglandulaire | forte | peu | peu noirci | clair |
| Lésion | forte | peu | peu noirci | **blanc** |

La grandeur physique standardisée qui mesure ce noircissement est la **densité optique** :

$$\mathrm{OD} = \log_{10}\!\left(\frac{I_0}{I_t}\right)$$

où $I_0$ est la lumière incidente d'un densitomètre et $I_t$ la lumière transmise à travers le film. **OD élevée = film noir = beaucoup de rayons X passés = tissu peu dense.**

Note l'inversion : **le tissu dense a une OD basse et s'affiche blanc**. Cette inversion est la source de la moitié des bugs d'affichage en imagerie sur film. Retiens-la.

Le DDSM documente, **pour chaque scanner**, la formule de conversion des valeurs brutes en OD [établi — c'est ce que décrit Lee et al. 2017]. CBIS-DDSM a appliqué ces formules, obtenu des OD en 64 bits, puis re-mappé vers du **niveau de gris 16 bits** (TIFF, puis DICOM). Les auteurs appliquent aussi le **seuillage 0,05–3,0** du DDSM pour la réduction de bruit, avec un drapeau pour le désactiver et retrouver les OD brutes [établi].

> **Ce que je ne sais pas et que tu dois vérifier toi-même** [incertain] : si le re-mappage OD → 16 bits a conservé ou inversé le sens. Je ne vais pas deviner. La vérification tient en trois lignes et c'est le tout premier truc à faire après le téléchargement :
>
> 1. Charge une mammographie complète.
> 2. Regarde un coin de l'image, **hors du sein** : c'est de l'air.
> 3. L'air doit s'afficher **noir**. Si la valeur du fond est proche du **minimum**, la convention est « min = noir » (`MONOCHROME2`). Si elle est proche du **maximum**, c'est « min = blanc » (`MONOCHROME1`) et il faut inverser avant tout traitement.
>
> Fais-le sur les deux sources : le DICOM TCIA **et** le JPEG Kaggle. Rien ne garantit qu'elles s'accordent.

### 3.2 Le DICOM n'est pas un format d'image

C'est une erreur d'entrée quasi universelle. **DICOM est un modèle d'objet**, pas un conteneur de pixels. Un fichier DICOM = un **en-tête** de métadonnées structurées + les données de pixels.

Hiérarchie [établi] :

```
Patient  →  Study (un examen)  →  Series (une acquisition)  →  Instance (une image)
```

Les tags qui décident du résultat de ton preprocessing :

| Tag | Rôle | Ce qui casse si tu l'ignores |
|---|---|---|
| `PhotometricInterpretation` | `MONOCHROME1` = valeur min affichée **blanche** ; `MONOCHROME2` = valeur min affichée **noire** | image inversée, sans erreur ni avertissement |
| `BitsAllocated` / `BitsStored` | 16 alloués, souvent 12 ou 14 réellement utilisés | tu normalises sur 65535 alors que le max réel est 4095 → image quasi noire |
| `Rows` / `Columns` | dimensions | appariement image/masque |
| `PixelSpacing` | **taille physique d'un pixel, en mm** | toute distance en mm est fausse — voir 3.5 |
| `WindowCenter` / `WindowWidth` | fenêtre d'affichage recommandée | contraste |
| `RescaleSlope` / `RescaleIntercept` | transformation affine vers l'unité physique | valeurs dans la mauvaise unité |
| `SeriesDescription` | texte libre : « full mammogram images », « ROI mask images », « cropped images » | **c'est ta clé de jointure** — voir 4 |

`MONOCHROME1` mérite qu'on s'y arrête : c'est exactement l'inversion de la section 3.1, encodée dans le standard. Une bibliothèque qui lit les pixels sans lire ce tag te rend une image à l'envers, silencieusement.

### 3.3 Le fenêtrage : 16 bits → 8 bits

Un écran affiche 256 niveaux ; ton image en a jusqu'à 65 536. Il faut choisir lesquels. Le radiologue le fait avec une **fenêtre** — un centre $c$ et une largeur $w$ :

$$
v_{\text{affiché}} =
\begin{cases}
0 & \text{si } v < c - w/2\\
255 & \text{si } v > c + w/2\\
255\cdot\dfrac{v - (c - w/2)}{w} & \text{sinon}
\end{cases}
$$

Tout ce qui sort de la fenêtre est **écrasé**, définitivement.

C'est là que se joue une bonne partie de ton preprocessing [inférence] :

- une **normalisation min-max naïve** sur l'image entière laisse quelques pixels extrêmes (marqueurs métalliques, étiquettes du scanner, bord du film) fixer les bornes, et tasse tout le tissu mammaire dans une poignée de niveaux ;
- **CLAHE** (*Contrast Limited Adaptive Histogram Equalization*) égalise l'histogramme par tuiles avec un plafond sur l'amplification, ce qui évite d'exploser le bruit dans les zones uniformes [établi]. C'est le choix par défaut en mammographie [consensus].

### 3.4 Ce qu'est réellement un masque

Le masque de ROI est une image binaire **de mêmes dimensions que la mammographie complète**, où les pixels de la lésion valent 1 (ou 255) et le reste 0 [établi]. Ce n'est pas un polygone, pas un contour, pas un RLE : une image pleine.

Trois artefacts par anomalie, distribués comme des séries DICOM distinctes :

1. la **mammographie complète** ;
2. le **masque ROI**, même taille ;
3. l'**imagette recadrée** (*crop*), découpée sur la boîte englobante de la ROI.

**À vérifier systématiquement** [inférence] : que `Rows`/`Columns` du masque correspondent bien à ceux de la mammographie associée, et que le fichier étiqueté « masque » est bien binaire et non une imagette en niveaux de gris. Un masque de dimensions différentes de son image est un signal d'appariement raté — pas quelque chose à redimensionner en silence pour que ça passe.

### 3.5 L'échelle physique

`PixelSpacing` donne la taille d'un pixel en millimètres. Les scanners du DDSM n'ont pas tous la même résolution [établi — Lee et al. signalent des scanners et résolutions différents selon l'établissement ; je ne donne pas les valeurs exactes de mémoire, **[incertain]**, elles sont dans les en-têtes].

Deux conséquences dures :

1. **Une distance en pixels n'est pas une distance physique.** Une lésion de 200 pixels ne fait pas la même taille réelle selon le scanner. Si tu veux rapporter un HD95 « en mm » (module 3), il faut multiplier par `PixelSpacing` — et cette valeur diffère d'une image à l'autre.
2. **Redimensionner change l'échelle physique.** Un `resize` à 512×512 appliqué à deux images de résolutions différentes ne les met pas à la même échelle physique — il les met à la même *taille en pixels*, ce qui n'est pas pareil. Pour un réseau qui doit apprendre « une microcalcification fait 0,5 mm », c'est destructeur [inférence].

### 3.6 Le miroir Kaggle

Le jeu `awsaf49/cbis-ddsm-breast-cancer-image-dataset` est une conversion **JPEG** des DICOM, avec les CSV à côté [établi].

Ce que la conversion coûte :

- **8 bits au lieu de 16** : la dynamique est écrasée d'un facteur 256. La chaîne de curation de 2017 avait précisément pour but de préserver cette dynamique sans perte ;
- **compression avec perte** : le JPEG travaille par blocs de 8×8 en DCT et lisse les hautes fréquences. Or une microcalcification **est** une haute fréquence, et un spicule aussi [inférence]. C'est exactement le signal du module 1 ;
- **plus de tags DICOM** : ni `PhotometricInterpretation`, ni `PixelSpacing`. La convention d'inversion est figée dans les pixels, et l'échelle physique est perdue.

Ce n'est pas une raison de ne pas l'utiliser — 163 Go de DICOM contre quelques Go de JPEG, l'arbitrage est réel. C'est une raison de **savoir ce que tu paies**, et de le dire à l'oral.

---

## 4. Sur tes données

### Le vrai travail de l'étape 1 : la jointure

Le problème concret, et il n'est pas anodin : **les colonnes de chemin des CSV pointent vers l'arborescence TCIA d'origine, pas vers les dossiers du miroir JPEG.** Les CSV donnent des colonnes du type `image file path`, `cropped image file path`, `ROI mask file path` [établi — documentées dans Lee et al.], contenant des chemins TCIA. Le miroir, lui, range les images dans des dossiers nommés par des identifiants opaques.

Le pont est `csv/dicom_info.csv`, qui porte pour chaque fichier converti ses métadonnées DICOM d'origine — dont `SeriesDescription` et le chemin. Le schéma de jointure :

```
mass_case_description_train_set.csv
        │  (chemin TCIA)
        ▼
   dicom_info.csv  ──  SeriesDescription ∈ {full mammogram, ROI mask, cropped}
        │
        ▼
   jpeg/<dossier>/<fichier>.jpg
```

> Les **noms exacts** des colonnes de `dicom_info.csv` sont **[incertain]** — je n'ai pas le fichier sous la main. Première chose à faire après téléchargement : un `head` sur chaque CSV, et écrire la jointure à partir de ce que tu vois, pas à partir de ce que je décris.

### Le piège de fuite, en chiffres

La fiche TCIA est explicite [établi] :

> 1 566 participants réels — mais **6 671 identifiants patient** dans les métadonnées DICOM, parce que chaque participant en possède plusieurs.

Traduis : **grouper par `PatientID` DICOM ne suffit pas.** Deux identifiants distincts peuvent désigner la même femme. Il faut remonter à l'identifiant de patiente que portent les CSV — la colonne `patient_id`, décrite comme « les 7 premiers caractères des images du dossier de cas » [établi, Lee et al.].

Et rappelle-toi la structure :

- une patiente → deux seins → deux incidences chacun (CC, MLO) → jusqu'à 4 images ;
- une image → potentiellement plusieurs anomalies ;
- une anomalie → 3 artefacts (image, masque, imagette) ;
- les jeux masse et calcification **se recouvrent** : certains cas portent les deux [établi].

Un `train_test_split(shuffle=True)` sur la table des anomalies met donc la vue CC et la vue MLO du même sein de part et d'autre de la frontière. Ton modèle voit la lésion à l'entraînement et la « retrouve » au test.

### Découpage correct

1. Le découpage train/test **officiel** est déjà fait, stratifié sur le BI-RADS, séparément pour masse et calcification [établi]. **On n'y touche pas** : c'est ce qui rend tes chiffres comparables.
2. La validation se taille **dans le train uniquement**, groupée par `patient_id`.
3. Vérification à écrire une fois et à garder : `set(patient_id_train) ∩ set(patient_id_val) == ∅`, idem avec le test.

---

## 5. Là où ça casse

**1. L'inversion silencieuse.** `MONOCHROME1` non lu → image à l'envers, aucun message d'erreur. Le modèle apprend quand même quelque chose, moins bien, et tu ne sauras jamais pourquoi. Test de fumée : le fond doit être noir.

**2. Normaliser sur `BitsAllocated`.** Diviser par 65535 quand `BitsStored` vaut 12 donne une image dont tout le contenu tient dans les 6 % inférieurs de la dynamique.

**3. Min-max sur l'image entière.** Une étiquette de scanner ou un marqueur métallique fixe le maximum et écrase tout le tissu. Il faut normaliser **après** avoir isolé la région mammaire, pas avant.

**4. Redimensionner avant de réfléchir.** Un `resize(512, 512)` sur une image de 3000×5000 divise les dimensions par ~6 à ~10. Une microcalcification de 0,5 mm disparaît purement et simplement, et un spicule aussi. Pour les masses c'est tolérable ; pour les calcifications c'est une suppression de la cible [inférence].

**5. Interpoler le masque.** Si tu redimensionnes, le masque doit l'être en **plus proche voisin**, jamais en bilinéaire : une interpolation lisse produit des valeurs intermédiaires qui ne sont plus binaires, et arrondit les contours — encore le fil rouge du module 1.

**6. Croire que `PatientID` = patiente.** 6 671 identifiants pour 1 566 femmes.

**7. Réparer un désaccord de dimensions par un `resize`.** Si masque et image n'ont pas la même taille, c'est un bug d'appariement. Le redimensionnement le camoufle et produit un masque décalé — un label faux, propre en apparence.

**8. Mélanger les deux sources.** Entraîner sur du JPEG 8 bits et évaluer sur du DICOM 16 bits (ou l'inverse) mesure surtout l'écart entre les deux encodages.

---

## 6. Contrôle

**a)** Tu charges une mammographie du miroir JPEG, tu constates que le fond hors du sein est **blanc**. Un camarade propose `img = 255 - img` et passe à la suite. Cette correction est-elle suffisante ? Dis précisément ce qu'elle répare, et quelle information perdue à la conversion DICOM → JPEG elle ne restaure pas — et pourquoi cette information manquante a une conséquence sur ta métrique de frontière, pas seulement sur l'affichage.

**b)** Ton pipeline redimensionne toutes les images en 1024×1024, masques inclus, avec la même interpolation bilinéaire. Tu obtiens un Dice de 0,81 sur les masses. Un relecteur affirme que ce chiffre est optimiste **et** pessimiste à la fois, pour deux raisons différentes. Trouve les deux : l'une vient du traitement du masque, l'autre de l'échelle physique de la section 3.5.

**c)** Tu écris ta jointure via `dicom_info.csv` et tu obtiens 9 812 appariements image/masque valides, alors que la somme des anomalies des quatre CSV en annonce davantage. Ton binôme propose de laisser tomber les manquants : « moins de 3 %, c'est du bruit ». Donne deux hypothèses **distinctes** sur l'origine de ces manquants — une bénigne, une qui invaliderait ton évaluation — et dis, pour chacune, quel contrôle sur les colonnes des CSV permettrait de trancher.
