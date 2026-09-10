# Module 4 — Le paysage des modèles

*Niveau : praticien. Prérequis : modules 1 à 3.*

---

## 1. Ancrage

Choisir une architecture, ce n'est pas choisir un score : c'est choisir **quelle hypothèse on fait sur la structure du problème** — et, ici, décider quelles informations spatiales survivent au passage dans le réseau.

---

## 2. Intuition

Le problème central de ce projet tient en deux nombres : l'image fait ~3000 × 5000 pixels, la lésion fait ~400 pixels de diamètre, et le détail qui porte le diagnostic — un spicule — fait quelques pixels de large.

Aucun réseau ne digère 15 millions de pixels à pleine résolution. Il faut donc **jeter de l'information**, et toute l'ingénierie consiste à choisir laquelle. Trois familles de réponses, et elles ne sont pas équivalentes :

| Stratégie | Ce qu'on garde | Ce qu'on perd |
|---|---|---|
| **Redimensionner** l'image entière | le contexte global, la position dans le sein | la haute fréquence : spicules, microcalcifications |
| **Découper** en imagettes | la résolution native, donc les détails fins | le contexte : « où suis-je dans le sein ? » |
| **Deux étages** : détecteur grossier puis segmenteur fin | les deux | de la complexité, et les erreurs du 1er étage sont définitives |

L'analogie : lire un journal. Redimensionner, c'est reculer de trois mètres — tu vois la mise en page, plus les mots. Découper, c'est lire à la loupe — tu vois les lettres, plus quel article tu lis. Le deux-étages, c'est repérer l'article de loin puis approcher la loupe : c'est ce que fait un radiologue, et ce que fait Shen 2019.

---

## 3. Le mécanisme

### 3.1 Le champ réceptif, la contrainte que tout le monde oublie

Le **champ réceptif** d'un neurone de sortie est la région de l'image d'entrée qui influence sa valeur. Pour une pile de $L$ convolutions 3×3 sans sous-échantillonnage, il vaut $2L+1$ pixels — croissance **linéaire**, donc très lente.

Règle pratique [consensus] : **le champ réceptif doit être au moins de la taille de l'objet à segmenter, et de préférence nettement plus grand**, pour que le réseau voie le contexte autour de la lésion.

Trois façons de l'agrandir, avec trois coûts distincts :

1. **empiler des couches** → coût en paramètres et en calcul, croissance linéaire ;
2. **sous-échantillonner** (pooling, stride) → croissance exponentielle, mais **perte de résolution spatiale** ;
3. **convolutions dilatées** (*atrous*) → on écarte les points d'échantillonnage du noyau d'un facteur $r$, le champ réceptif croît comme $r$ **sans perdre de résolution ni ajouter de paramètres** [établi].

Toute la zoologie ci-dessous n'est qu'une manière différente de trancher ce compromis champ réceptif / résolution.

### 3.2 U-Net : pourquoi les connexions résiduelles latérales sont le sujet

Structure : un **encodeur** qui sous-échantillonne (le champ réceptif grandit, la résolution meurt), un **décodeur** qui sur-échantillonne, et — c'est là tout l'intérêt — des **connexions latérales** (*skip connections*) qui réinjectent, à chaque niveau du décodeur, la carte de caractéristiques de même résolution venue de l'encodeur.

**Pourquoi c'est exactement l'architecture qu'il faut ici.** Le passage dans l'encodeur détruit la précision spatiale : après quatre poolings, un pixel de la carte profonde correspond à 16 pixels d'entrée. Un décodeur seul ne peut pas reconstruire un contour qu'il ne voit plus — il produit des masques ronds et lisses. Les connexions latérales **réinjectent la haute fréquence**.

Autrement dit : les *skip connections* sont le mécanisme qui préserve la **marge**. Le fil rouge des modules 1 et 3 a une contrepartie architecturale, et c'est celle-là.

**Variantes :**
- **U-Net++** : densifie les connexions latérales par des blocs de convolution imbriqués, pour réduire l'écart sémantique entre encodeur et décodeur.
- **ResUNet** : blocs résiduels dans l'encodeur → plus profond sans dégradation du gradient.
- **Attention U-Net** : place une porte d'attention sur chaque connexion latérale, qui **pondère spatialement** ce qui est réinjecté. Motivation directe sur mammographie : la quasi-totalité de l'image est du fond, et les *skips* réinjectent ce fond aussi fidèlement que la lésion [inférence].

### 3.3 DeepLabv3+ : le champ réceptif sans le sacrifice

Repose sur les convolutions dilatées (§ 3.1). Le module **ASPP** (*Atrous Spatial Pyramid Pooling*) applique **en parallèle** plusieurs taux de dilatation, puis concatène : le réseau voit simultanément à plusieurs échelles.

Attention au *output stride* : DeepLab produit typiquement une carte à 1/8 ou 1/16 de la résolution d'entrée, puis sur-échantillonne. **Ce sur-échantillonnage final lisse les contours** [inférence] — même problème que ci-dessus, moins bien traité que par les *skips* d'un U-Net.

### 3.4 nnU-Net : ce n'est pas une architecture

C'est une **méthode de configuration automatique**. À partir d'une « empreinte » du jeu de données (tailles d'images, espacement des pixels, distribution d'intensité, nombre de classes), elle dérive par des règles fixes : le prétraitement, la taille des patches, la taille du batch, la profondeur du réseau, le schéma d'augmentation, la stratégie d'entraînement.

Empiriquement c'est la **référence la plus difficile à battre** en segmentation médicale — elle a remporté le Medical Segmentation Decathlon [établi].

**Conseil stratégique pour ton projet** [inférence] : ne consacre pas ton semestre à essayer de battre nnU-Net. Utilise-la comme **plancher** — « voilà ce qu'on obtient sans réfléchir » — et investis ton temps dans l'analyse stratifiée du module 3, qui est ce que la grille d'évaluation MLB récompense (« explications sur la pratique, validation », « résultats obtenus »).

### 3.5 Mask R-CNN et la famille détection d'abord

Chaîne : un **RPN** (*Region Proposal Network*) propose des boîtes candidates → **ROI-Align** extrait une grille de caractéristiques par boîte → trois têtes en parallèle : classe, régression de boîte, **masque**.

**Ce que ça t'apporte, et que le U-Net ne donne pas gratuitement :**
- la séparation en **instances** — deux lésions voisines restent deux objets, là où un U-Net produit une seule composante connexe ;
- un **score par lésion**, donc une courbe FROC directement calculable (module 3, § 3.6).

**Ce que ça te coûte :** la tête de masque produit une grille de faible résolution — **28 × 28** dans l'implémentation standard [établi] — ensuite étirée à la taille de la boîte. Sur une masse de 400 pixels, un spicule est irrécupérable [inférence]. Excellent pour la détection, structurellement médiocre pour la finesse du contour.

### 3.6 Transformers

L'auto-attention donne un **champ réceptif global dès la première couche** — le problème du § 3.1 disparaît. Deux coûts :

1. **complexité quadratique** en nombre de tokens ; sur des images médicales grandes, c'est prohibitif. **Swin** y répond par une attention par fenêtres décalées, de coût linéaire ;
2. **appétit en données** : sans le biais inductif de localité de la convolution, il faut beaucoup plus d'exemples [consensus].

**Contrainte de résolution à voir** : un ViT découpe l'image en patches (16 × 16 typiquement). Cette taille de patch fixe un **plancher de résolution spatiale** dans le tronc du réseau. **TransUNet** contourne le problème en gardant un encodeur convolutif et en plaçant le transformer au niveau le plus profond, avec les *skips* du U-Net conservés.

Sur ton volume de données (§ 4), un transformer entraîné de zéro n'a aucune chance. Avec un pré-entraînement massif, c'est jouable.

### 3.7 Modèles de fondation — et le piège du prompt

**SAM** segmente à la demande : on lui fournit un **prompt** (un point, une boîte, un masque grossier) et il rend le masque de l'objet désigné. **MedSAM** en est un ré-entraînement sur imagerie médicale.

> **Le piège, et il est éliminatoire.** Si tu utilises la **boîte englobante de la vérité terrain** comme prompt, tu as donné au modèle la réponse à la question que tu prétends poser. Tu ne mesures plus « sait-il trouver la lésion ? » mais « à qui je dis où est la lésion, sait-il en tracer le contour ? ».
>
> Ce n'est pas nul — c'est une **tâche légitime**, l'affinage de contour, et elle a même du sens ici vu la qualité des masques (module 1). Mais ce n'est **pas** de la détection, et les chiffres ne se comparent à aucun article de détection. Si tu explores cette piste, écris la phrase « le prompt provient de la vérité terrain » en toutes lettres sur ta slide [inférence].

### 3.8 Les fonctions de coût

| Loss | Formule / idée | Ce qu'elle règle |
|---|---|---|
| **BCE** | $-\big(y\log p + (1-y)\log(1-p)\big)$ par pixel | gradient stable, mais **dominée par le fond** quand les positifs sont rares |
| **Dice loss** | $1 - \mathrm{Dice}$ (version continue) | insensible au déséquilibre, car sans TN — mais gradient instable sur petits objets |
| **Tversky** | $\dfrac{TP}{TP + \alpha FP + \beta FN}$ | $\beta > \alpha$ pénalise davantage les **FN** → remonte la sensibilité, au prix du FPPI |
| **Focal** | $-(1-p)^{\gamma}\log p$ | écrase le poids des exemples **faciles** → concentre l'apprentissage sur les pixels ambigus, c'est-à-dire les **contours** |

En pratique, **Dice + BCE** est le mélange de référence [consensus] : le BCE donne un gradient exploitable en début d'entraînement, le Dice porte l'objectif final.

**Le lien avec le module 3** : la loss est ta métrique, en version dérivable. Optimiser une loss purement volumique **récompense le lissage des contours**. Si le contour t'importe, il faut soit un terme de frontière explicite (*boundary loss*), soit une loss focale qui surpondère les pixels ambigus — qui sont majoritairement les pixels de bord [inférence].

### 3.9 Transfert et pré-entraînement

Le pré-entraînement ImageNet fonctionne sur les images médicales alors même qu'une mammographie n'a rien d'une photo de chat [consensus] : les premières couches apprennent des bords, des textures, des orientations, qui sont universels.

**La recette Shen 2019** [établi] mérite d'être comprise, c'est le meilleur usage possible de ton jeu :

1. entraîner un **classifieur d'imagettes** sur les crops de ROI — beaucoup d'exemples, tâche facile, supervision dense ;
2. **convertir** ce classifieur en modèle d'image entière, en remplaçant la tête et en réutilisant tout le tronc.

L'étape 1 fabrique un encodeur qui sait déjà à quoi ressemble une lésion, avant même de voir une image complète. Transposée à ton problème : **entraîne d'abord ton U-Net sur les imagettes ROI, puis réutilise son encodeur pour le passage à l'image entière** [inférence, transposition de la recette]. C'est ce que fait le pipeline étape 4 → étape 5 du schéma du projet.

---

## 4. Sur tes données

### 4.1 Le volume réel

Anomalies de type masse [établi, Table 3 de Lee et al.] :

- train : 387 (bénignes) + 361 (malignes) = **748 anomalies**
- test : 135 + 87 = **222 anomalies**

Chaque anomalie est vue sur CC et/ou MLO, donc au plus ~1 500 paires image/masque en train [inférence]. Après avoir taillé une validation groupée par patiente, il reste de l'ordre de **600 anomalies d'entraînement**.

**Ce nombre décide de tout :**

| Volume | Conséquence |
|---|---|
| ~600 anomalies | transformer entraîné de zéro : exclu |
| | encodeur pré-entraîné : obligatoire |
| | augmentation agressive : obligatoire |
| | validation croisée plutôt qu'un split unique : fortement conseillé [inférence] |

### 4.2 Quelles architectures préservent la marge

Le tableau qui n'est jamais dans les articles, et qui est le vrai critère de ton projet :

| Architecture | Résolution de sortie | Mécanisme de préservation du contour | Verdict sur la marge |
|---|---|---|---|
| U-Net | pleine | *skip connections* | **bon** |
| Attention U-Net | pleine | *skips* pondérés spatialement | bon |
| DeepLabv3+ | 1/8 ou 1/16 puis interpolation | ASPP, pas de *skip* haute résolution complet | moyen — lisse |
| Mask R-CNN | 28 × 28 par boîte, étiré | aucun | **mauvais** |
| ViT pur | plancher = taille du patch | aucun | mauvais |
| TransUNet | pleine | *skips* du U-Net conservés | bon |
| nnU-Net | pleine | U-Net configuré automatiquement | bon |

[inférence sur la colonne « verdict » : conséquence des résolutions de sortie, pas un résultat mesuré sur CBIS-DDSM.]

### 4.3 Plan minimal défendable

1. **U-Net, encodeur ResNet pré-entraîné, sur imagettes ROI de masses**, loss Dice+BCE. C'est la référence de base, étape 4 du pipeline. Objectif ≈ 0,75–0,80 de Dice, c'est-à-dire au niveau de la qualité des labels eux-mêmes (module 1).
2. **nnU-Net sur les mêmes données**, sans réglage. Si tu ne bats pas nnU-Net, tu le dis — c'est un résultat honnête et fréquent.
3. **Une variation, une seule, motivée** : Attention U-Net, ou une loss avec terme de frontière. L'intérêt n'est pas de gagner 2 points de Dice (module 3, § 4.2 : ce ne serait pas significatif) mais de **montrer un effet sur le HD95**.
4. **L'analyse stratifiée** — par densité, subtilité, pathologie. C'est là qu'est la note.

---

## 5. Là où ça casse

**1. Le prompt de SAM issu de la vérité terrain.** § 3.7. C'est la faute la plus facile à commettre et la plus fatale à l'oral.

**2. Redimensionner puis se plaindre des contours.** Si tu passes de 3000×5000 à 512×512, tu as divisé la résolution par ~7 : le spicule n'est plus dans l'entrée. Aucune architecture ne récupère un signal absent.

**3. Le déséquilibre au niveau de l'imagette.** Si tu découpes l'image en tuiles, l'écrasante majorité ne contient aucune lésion. Un échantillonnage uniforme entraîne un modèle qui prédit « vide » ; un échantillonnage centré sur les lésions produit un modèle qui n'a jamais vu de tissu normal et qui hallucine à l'inférence sur image entière [inférence]. Le ratio d'échantillonnage est un hyperparamètre, pas un détail.

**4. Comparer ton Dice à celui d'un article.** Prétraitement différent, définition du Dice différente (moyen vs global, module 3), sous-ensemble différent, cas vides traités différemment. Le seul chiffre comparable est celui que tu calcules toi-même sur le **test officiel** avec un protocole écrit.

**5. Vouloir battre nnU-Net.** C'est le trou noir des projets de segmentation. § 3.4.

**6. La TTA (*test-time augmentation*) non déclarée.** Moyenner les prédictions sur plusieurs versions augmentées de l'image gagne quelques points, presque gratuitement. C'est légitime — à condition de le dire, sinon tu compares un modèle + TTA à un modèle nu.

**7. Un seul split.** Avec 600 anomalies d'entraînement et les intervalles de confiance du module 3, un split unique produit un classement de modèles qui change si tu changes la graine aléatoire.

**8. Traiter masses et calcifications avec le même modèle à la même échelle.** Facteur 40 en taille (module 1). Ce n'est pas un problème de capacité, c'est un problème de résolution d'entrée.

---

## 6. Contrôle

**a)** Tu entraînes un U-Net sur imagettes ROI recadrées autour des masses. Le Dice de validation atteint 0,84. Tu appliques ensuite le même modèle, tel quel, sur des mammographies complètes découpées en tuiles : le Dice s'effondre et le modèle produit des dizaines de fausses détections par image. Donne **deux causes structurelles distinctes** à cet effondrement — l'une liée à ce que le modèle a vu à l'entraînement, l'autre au champ réceptif du § 3.1 — et dis, pour chacune, si la recette Shen du § 3.9 la corrige ou non.

**b)** Un camarade propose de remplacer la loss `Dice + BCE` par une Tversky avec $\alpha = 0{,}3$, $\beta = 0{,}7$, en argumentant qu'« en cancérologie il vaut mieux un faux positif qu'un cancer manqué ». L'argument clinique est juste. Explique pourquoi la conclusion technique ne suit pas automatiquement, en te servant du FPPI du module 3 — et dis quelle courbe, plutôt qu'un choix de $\beta$, répond réellement à la question posée.

**c)** Tu obtiens : U-Net 0,79 de Dice, nnU-Net 0,83, Mask R-CNN 0,71. Ton binôme conclut que nnU-Net est le meilleur modèle pour ce projet et propose de ne garder que lui. En mobilisant le module 3 **et** le tableau du § 4.2, donne deux raisons indépendantes pour lesquelles ce classement ne suffit pas à trancher — et dis quel chiffre supplémentaire pourrait faire remonter Mask R-CNN en tête malgré son Dice le plus bas.
